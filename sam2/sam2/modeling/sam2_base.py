# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from loguru import logger

import numpy as np
import os
import json
import torch
import torch.distributed
import torch.nn.functional as F
import cv2

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

from sam2.utils.reselect_metrics import compute_acceleration_rmse
from sam2.utils.kalman_filter import KalmanFilter
from sam2.utils.motion_predictor.predictor import MotionPredictor
from sam2.utils.motion_predictor.predictor import xyxy_to_xywh, xyxy_to_xcycwh, xyxy_to_x1y1ah, xyxy_to_xcycah, convert_bbox_to_xyxy, \
    compute_iou, compute_shape_score, compute_size_score, compute_center_distance_score

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # Whether to use SAMURAI or original SAM 2
        samurai_mode: bool = False,
        # Hyperparameters for SAMURAI
        stable_frames_threshold: int = 15,
        stable_ious_threshold: float = 0.3,
        min_obj_score_logits: float = -1,
        kf_score_weight: float = 0.15,
        memory_bank_iou_threshold: float = 0.5,
        memory_bank_obj_score_threshold: float = 0.0,
        memory_bank_kf_score_threshold: float = 0.0,
        memory_bank_init_feat_similarity_threshold: float = 0.0,
        memory_bank_last_feat_similarity_threshold: float = 0.0,

        # Modifications on memory bank selection
        # Whether to apply sigmoid to the object score logits
        apply_sigmoid_to_obj_score_logits: bool = False,
        memory_selection_strategy: str = "backward", # "backward" or "topk"
        memory_selection_range: int = 30, # Range of the most recent k frames to consider for top-k selection
        memory_selection_sam2iou_weight: float = 1.0,
        memory_selection_obj_score_weight: float = 0.0,
        memory_selection_kf_score_weight: float = 0.0,
        memory_selection_init_feat_similarity_weight: float = 0.0,
        memory_selection_last_feat_similarity_weight: float = 0.0,
        
        # Whether to use Motion Predictor or original SAM 2
        samosa_mode: bool = False,
        # Hyperparameters for Motion Predictor
        mp_model_type: str = "MLPMarkovModel",
        state_type: str = "xywh", # "xywh" or "xcycwh" or "x1y1ah" or "xcycah"
        mp_state_dim: int = 8,
        mp_hidden_dim: int = 64,
        mp_history_length: int = 1,
        rnn_layers: int = 1,
        normalize_box_size: bool = False,
        normalize_box_pos: bool = False,
        short_mem_length: int = 1,
        sam2_iou_score_weight: float = 0.85,
        mp_iou_score_weight: float = 0.15,
        mp_shape_score_weight: float = 0.0,
        mp_size_score_weight: float = 0.0,
        mp_center_distance_score_weight: float = 0.0,

        # Whether to activate error detection mode
        error_detection_mode: bool = False,
        ignore_error_memories: bool = False,
        crop_feature_by_mask: bool = True,
        restrict_max_error_memory_num_in_pool: int = 30,
        # Hyperparameters for error detection
        error_detect_adjacent_frames_iou_threshold: float = 0.0,
        error_detect_adjacent_frames_feature_similarity_threshold: float = 0.0,
        error_detect_shape_acc_threshold: float = 0.0,
        error_detect_size_acc_threshold: float = 0.0,
        error_detect_strategy: str = "or", # "or" or "and"
        error_detect_recovery_history_length: int = 5,
        # Hyperparameters for error recovery
        update_only_exist_box_for_error_recovery: bool = True,
        error_recovery_feature_similarity_threshold: float = 0.0,
        error_recovery_shape_similarity_threshold: float = 0.0,
        error_recovery_size_similarity_threshold: float = 0.0,
        recover_at_best_iou: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Whether to use SAMURAI or original SAM 2
        self.samurai_mode = samurai_mode

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0

        # Debug purpose
        self.history = {} # debug
        self.frame_cnt = 0 # debug

        # Hyperparameters for SAMURAI
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold
        self.memory_bank_init_feat_similarity_threshold = memory_bank_init_feat_similarity_threshold
        self.memory_bank_last_feat_similarity_threshold = memory_bank_last_feat_similarity_threshold

        print(f"\033[93mSAMURAI mode: {self.samurai_mode}\033[0m")

        print(f"\033[93mmemory bank iou threshold: {self.memory_bank_iou_threshold}\033[0m")
        print(f"\033[93mmemory bank obj score threshold: {self.memory_bank_obj_score_threshold}\033[0m")
        print(f"\033[93mmemory bank kf score threshold: {self.memory_bank_kf_score_threshold}\033[0m")
        print(f"\033[93mmemory bank init feat similarity threshold: {self.memory_bank_init_feat_similarity_threshold}\033[0m")
        print(f"\033[93mmemory bank last feat similarity threshold: {self.memory_bank_last_feat_similarity_threshold}\033[0m")

        # Modifications on memory bank selection
        self.apply_sigmoid_to_obj_score_logits = apply_sigmoid_to_obj_score_logits
        self.memory_selection_strategy = memory_selection_strategy
        assert self.memory_selection_strategy in ["backward", "topk"], "Memory selection strategy must be either [backward] or [topk]"
        self.memory_selection_range = memory_selection_range
        assert self.memory_selection_range > 5, "Memory selection range must be greater than 5"
        self.memory_selection_sam2iou_weight = memory_selection_sam2iou_weight
        self.memory_selection_obj_score_weight = memory_selection_obj_score_weight
        self.memory_selection_kf_score_weight = memory_selection_kf_score_weight
        self.memory_selection_init_feat_similarity_weight = memory_selection_init_feat_similarity_weight
        self.memory_selection_last_feat_similarity_weight = memory_selection_last_feat_similarity_weight
        print(f"\033[93mMemory selection strategy: {self.memory_selection_strategy}\033[0m")
        if self.memory_selection_strategy == "topk":
            print(f"\033[93mMemory selection range: {self.memory_selection_range}\033[0m")
            print(f"\033[93mMemory selection SAM2 IoU Weight: {self.memory_selection_sam2iou_weight}\033[0m")
            print(f"\033[93mMemory selection Object Score Weight: {self.memory_selection_obj_score_weight}\033[0m")
            print(f"\033[93mMemory selection KF Score Weight: {self.memory_selection_kf_score_weight}\033[0m")
            print(f"\033[93mMemory selection Initial Feature Similarity Weight: {self.memory_selection_init_feat_similarity_weight}\033[0m")
            print(f"\033[93mMemory selection Latest Feature Similarity Weight: {self.memory_selection_last_feat_similarity_weight}\033[0m")
        
        # Whether to use Motion Predictor or original SAM 2
        self.samosa_mode = samosa_mode
        print("")
        print(f"\033[93mSAMOSA mode: {self.samosa_mode}\033[0m")
        # Hyperparameters for Motion Predictor
        self.mp_state_dim = mp_state_dim
        self.mp_hidden_dim = mp_hidden_dim
        self.mp_history_length = mp_history_length
        self.mp_stable_ious_threshold = stable_ious_threshold
        self.sam2_iou_score_weight = sam2_iou_score_weight
        self.mp_iou_score_weight = mp_iou_score_weight
        self.mp_shape_score_weight = mp_shape_score_weight
        self.mp_size_score_weight = mp_size_score_weight
        self.mp_center_distance_score_weight = mp_center_distance_score_weight
        self.short_mem_length = short_mem_length

        if self.samosa_mode:
            print(f"\033[93mMP Model Type: {mp_model_type}\033[0m")
            print(f"\033[93mMP History Length: {self.mp_history_length}\033[0m")
            print(f"\033[93mMP Stable IOU Threshold: {self.mp_stable_ious_threshold}\033[0m")
            print(f"\033[93mSAM2 IoU Score Weight: {self.sam2_iou_score_weight}\033[0m")
            print(f"\033[93mMP IoU Score Weight: {self.mp_iou_score_weight}\033[0m")
            print(f"\033[93mMP Shape Score Weight: {self.mp_shape_score_weight}\033[0m")
            print(f"\033[93mMP Size Score Weight: {self.mp_size_score_weight}\033[0m")
            print(f"\033[93mMP Center Distance Score Weight: {self.mp_center_distance_score_weight}\033[0m")
            print(f"\033[93mShort Memory Length: {self.short_mem_length}\033[0m")
            print(f"\033[93mNormalize Box Size: {normalize_box_size}\033[0m")
            print(f"\033[93mNormalize Box Pos: {normalize_box_pos}\033[0m")

        if self.samosa_mode:
            self.motion_predictor = MotionPredictor(
                model_type=mp_model_type, state_type = state_type, state_dim=mp_state_dim, 
                hidden_dim=mp_hidden_dim, history_size=mp_history_length, rnn_layers=rnn_layers, 
                normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, reverse=False
            )
        self.state_type = state_type
        if self.state_type == "xywh":
            self.convert_xyxy_to_current_state = xyxy_to_xywh
        elif self.state_type == "xcycwh":
            self.convert_xyxy_to_current_state = xyxy_to_xcycwh
        elif self.state_type == "x1y1ah":
            self.convert_xyxy_to_current_state = xyxy_to_x1y1ah
        elif self.state_type == "xcycah":
            self.convert_xyxy_to_current_state = xyxy_to_xcycah
        else:
            raise ValueError(f"Unsupported state type: {self.state_type}")
        
        if self.samosa_mode or self.samurai_mode:
            print(f"\033[93mMP State Type: {self.state_type}\033[0m")
            
        # Whether to activate error detection mode
        self.error_detection_mode = error_detection_mode
        self.ignore_error_memories = ignore_error_memories
        self.crop_feature_by_mask = crop_feature_by_mask
        # self.restrict_max_error_memory_num_to_choose = restrict_max_error_memory_num_to_choose
        self.restrict_max_error_memory_num_in_pool = restrict_max_error_memory_num_in_pool
        self.error_detect_adjacent_frames_iou_threshold = error_detect_adjacent_frames_iou_threshold
        self.error_detect_adjacent_frames_feature_similarity_threshold = error_detect_adjacent_frames_feature_similarity_threshold
        self.error_detect_shape_acc_threshold = error_detect_shape_acc_threshold
        self.error_detect_size_acc_threshold = error_detect_size_acc_threshold
        self.error_detect_strategy = error_detect_strategy
        self.error_detect_recovery_history_length = error_detect_recovery_history_length
        self.update_only_exist_box_for_error_recovery = update_only_exist_box_for_error_recovery
        self.error_recovery_feature_similarity_threshold = error_recovery_feature_similarity_threshold
        self.error_recovery_shape_similarity_threshold = error_recovery_shape_similarity_threshold
        self.error_recovery_size_similarity_threshold = error_recovery_size_similarity_threshold
        self.recover_at_best_iou = recover_at_best_iou
        print("")
        print(f"\033[93mError Detection mode: {self.error_detection_mode}\033[0m")
        self.is_current_segmentation_likely_to_be_error = False
        if self.error_detection_mode:
            # Store historical mask and bbox information for error detection
            self.mask_bank_for_error_detect_recovery = []  # Store recent masks
            self.bbox_bank_for_error_detect_recovery = []  # Store recent bboxes [x1, y1, x2, y2]
            self.feature_bank_for_error_detect_recovery = []  # Store recent average features

            print(f"\033[93mIgnore Error Memories: {self.ignore_error_memories}\033[0m")
            print(f"\033[93mError Detect Strategy: {self.error_detect_strategy}\033[0m")
            # print(f"\033[93mRestrict Max Error Memory Num to Choose: {self.restrict_max_error_memory_num_to_choose}\033[0m")
            print(f"\033[93mRestrict Max Error Memory Num in Pool: {self.restrict_max_error_memory_num_in_pool}\033[0m")
            print(f"\033[93mError Detect Adjacent Frames IOU Threshold: {self.error_detect_adjacent_frames_iou_threshold}\033[0m")
            print(f"\033[93mError Detect Adjacent Frames Feature Similarity Threshold: {self.error_detect_adjacent_frames_feature_similarity_threshold}\033[0m")
            print(f"\033[93mError Detect Shape Acc Threshold: {self.error_detect_shape_acc_threshold}\033[0m")
            print(f"\033[93mError Detect Size Acc Threshold: {self.error_detect_size_acc_threshold}\033[0m")
            print(f"\033[93mError Exit History Length: {self.error_detect_recovery_history_length}\033[0m")
            print(f"\033[93mError Exit Feature Similarity Threshold: {self.error_recovery_feature_similarity_threshold}\033[0m")
            print(f"\033[93mError Exit Shape Similarity Threshold: {self.error_recovery_shape_similarity_threshold}\033[0m")
            print(f"\033[93mError Exit Size Similarity Threshold: {self.error_recovery_size_similarity_threshold}\033[0m")
            print(f"\033[93mError Exit at Best IoU: {self.recover_at_best_iou}\033[0m")
        # either samurai or samosa mode, not both
        assert not (self.samurai_mode and self.samosa_mode), "Either Kalman Filter or Markov Model for MP can be used, not both"
        
        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()
    
    def _compute_feature_similarity_score(self, average_history_feature, backbone_features, low_res_multibboxes, low_res_multimasks=None):
        scores = []
        cos = torch.nn.CosineSimilarity(dim=-1)
        
        if self.crop_feature_by_mask and low_res_multimasks is not None:
            # Feature extraction based on mask regions
            for i in range(len(low_res_multibboxes)):
                mask = low_res_multimasks[i]  # [H*4, W*4]
                # Binarize the mask: values greater than 0 become 1, values less than or equal to 0 become 0
                mask = (mask > 0).float()
                if mask.numel() == 0 or mask.sum() == 0:
                    scores.append(torch.tensor(0, device=self.device))
                    continue
                
                # Resize the mask to the backbone_features size
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(backbone_features.size(2), backbone_features.size(3)),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                # Perform weighted average pooling based on the mask
                masked_features = backbone_features.squeeze() * mask_resized.unsqueeze(0)  # [C, H, W]
                mask_sum = mask_resized.sum()
                
                if mask_sum > 0:
                    local_feature = masked_features.sum(dim=(1, 2)) / mask_sum  # [C]
                else:
                    local_feature = torch.zeros(backbone_features.size(1), device=self.device)
                
                scores.append(cos(local_feature, average_history_feature))
        else:
            # Feature extraction based on bounding boxes (original implementation)
            for i in range(len(low_res_multibboxes)):
                x1, y1, x2, y2 = low_res_multibboxes[i][0], low_res_multibboxes[i][1], low_res_multibboxes[i][2], low_res_multibboxes[i][3]
                local_feature = F.adaptive_avg_pool2d(backbone_features[:, :, y1:y2, x1:x2], (1, 1)).squeeze()
                if x1 == x2 or y1 == y2:
                    scores.append(torch.tensor(0, device=self.device))
                else:
                    scores.append(cos(local_feature, average_history_feature))
        
        score = torch.stack(scores)
        return score

    def _detect_error(self, current_mask, current_bbox, backbone_features, low_res_masks):
        """
        Error detection method.
        Detection criteria:
        1. The IoU of masks in two adjacent frames is below the threshold.
        2. The second-order difference in target box aspect ratio change is above the threshold.
        3. The second-order difference in target box area change is above the threshold (normalization required).
        4. The cosine similarity between the current and previous frame backbone features is below the threshold.
        
        Args:
            current_mask: Current frame mask [B, 1, H, W] or [1, H, W]
            current_bbox: Current frame bbox [x1, y1, x2, y2]
            backbone_features: Current frame backbone features [1, 256, 64, 64]
            low_res_masks: Current frame low_res_mask [1, 1, 256, 256]
        Returns:
            bool: True indicates that the current frame may contain an error.
        """
        # Return False if the history length is insufficient
        if len(self.mask_bank_for_error_detect_recovery) < 1:
            return False
        
        # Get the previous frame's mask and bbox
        prev_mask = self.mask_bank_for_error_detect_recovery[-1]
        if len(self.bbox_bank_for_error_detect_recovery) >= 2:
            prev_bbox = self.bbox_bank_for_error_detect_recovery[-1]
            prev_prev_bbox = self.bbox_bank_for_error_detect_recovery[-2]  # Bbox from two frames ago
        else:
            return False
        
        # Detection 1: IoU of masks in two adjacent frames (only when both frames have targets)
        is_IoU_error = False
        if self.error_detect_adjacent_frames_iou_threshold > 0:
            # Check whether both frames have targets
            both_frames_have_target = (current_bbox != [0, 0, 0, 0] and prev_bbox != [0, 0, 0, 0])
            
            if both_frames_have_target:
                # Compute IoU
                current_mask_binary = current_mask.detach().cpu() > 0.0
                prev_mask_binary = prev_mask > 0.0
                
                # Ensure consistent shapes
                if current_mask_binary.shape != prev_mask_binary.shape:
                    # If shapes are inconsistent, resize prev_mask
                    prev_mask_binary = torch.nn.functional.interpolate(
                        prev_mask_binary.float(),
                        size=current_mask_binary.shape[-2:],
                        mode='nearest'
                    ) > 0.0
                
                # Compute IoU
                intersection = (current_mask_binary & prev_mask_binary).sum().float()
                union = (current_mask_binary | prev_mask_binary).sum().float()
                iou = intersection / union if union > 0 else 0.0
                
                # # for debug only
                # print(f"IoU of adjacent frames: {iou}")
                if iou < self.error_detect_adjacent_frames_iou_threshold:
                    is_IoU_error = True
                    # print(f"⚠️Found possible error, IoU:{iou}")
        
        # Detections 2 and 3: at least 2 frames of data are needed to compute second-order differences
        is_Shape_acc_error = False
        is_Size_acc_error = False
        # Compute second-order differences only when 3 consecutive frames all have targets
        all_frames_have_target = (current_bbox != [0, 0, 0, 0] and 
                                    prev_bbox != [0, 0, 0, 0] and 
                                    prev_prev_bbox != [0, 0, 0, 0])
        
        if all_frames_have_target:
            # Compute the second-order difference of the aspect ratio
            if self.error_detect_shape_acc_threshold > 0:
                # Compute the current frame aspect ratio
                current_aspect = (current_bbox[2] - current_bbox[0]) / max(current_bbox[3] - current_bbox[1], 1)
                
                # Compute the previous frame aspect ratio
                prev_aspect = (prev_bbox[2] - prev_bbox[0]) / max(prev_bbox[3] - prev_bbox[1], 1)
                
                # Compute the aspect ratio from two frames ago
                prev_prev_aspect = (prev_prev_bbox[2] - prev_prev_bbox[0]) / max(prev_prev_bbox[3] - prev_prev_bbox[1], 1)
                
                # Compute the first-order difference
                first_diff = current_aspect - prev_aspect
                # Compute the second-order difference
                second_diff = first_diff - (prev_aspect - prev_prev_aspect)
                
                # # for debug only
                # print(f"Shape acc of adjacent frames: {second_diff}")
                if abs(second_diff) > self.error_detect_shape_acc_threshold:
                    is_Shape_acc_error = True
                    # print(f"⚠️Found possible error, Shape acc:{second_diff}")
        
            # Compute the second-order difference of area (normalized by the area from two frames ago)
            if self.error_detect_size_acc_threshold > 0:
                # Compute the actual area from two frames ago as the normalization baseline
                base_area = (prev_prev_bbox[2] - prev_prev_bbox[0]) * (prev_prev_bbox[3] - prev_prev_bbox[1])
                
                # Use the baseline area to normalize the areas of all frames
                current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1]) / max(base_area, 1.0)
                prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]) / max(base_area, 1.0)
                
                # Compute the area from two frames ago (normalized so the baseline area is 1)
                prev_prev_area = base_area / max(base_area, 1.0)  # = 1.0
                
                # Compute the first-order difference
                first_diff = current_area - prev_area
                # Compute the second-order difference
                second_diff = first_diff - (prev_area - prev_prev_area)
                
                # # for debug only
                # print(f"Size acc of adjacent frames: {second_diff}")
                if abs(second_diff) > self.error_detect_size_acc_threshold:
                    is_Size_acc_error = True
                    # print(f"⚠️Found possible error, Size acc:{second_diff}")

        # Detection 4: cosine similarity between current and previous frame backbone features is below the threshold
        is_Feature_error = False
        if self.error_detect_adjacent_frames_feature_similarity_threshold > 0:
            cos = torch.nn.CosineSimilarity(dim=-1)
            if self.crop_feature_by_mask and low_res_masks is not None:
                mask = low_res_masks  # [1, 1, 256, 256]
                # Binarize the mask: values greater than 0 become 1, values less than or equal to 0 become 0
                mask = (mask > 0).float()
                if not (mask.numel() == 0 or mask.sum() == 0):
                    # Resize the mask to the backbone_features size
                    mask_resized = F.interpolate(
                        mask.float(),
                        size=(backbone_features.size(2), backbone_features.size(3)),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    # Perform weighted average pooling based on the mask
                    masked_features = backbone_features.squeeze() * mask_resized.unsqueeze(0)  # [C, H, W]
                    mask_sum = mask_resized.sum()
                    
                    if mask_sum > 0:
                        local_feature = masked_features.sum(dim=(1, 2)) / mask_sum  # [C]
                    else:
                        local_feature = torch.zeros(backbone_features.size(1), device=self.device)
                    
                    average_history_feature = torch.mean(torch.stack(self.feature_bank_for_error_detect_recovery), dim=0)
                    feature_similarity = cos(local_feature, average_history_feature)
                    if feature_similarity < self.error_detect_adjacent_frames_feature_similarity_threshold:
                        is_Feature_error = True
                        # print(f"⚠️Found possible error, Feature similarity:{feature_similarity}")
            else:
                # Feature extraction based on bounding boxes (original implementation)
                non_zero_indices_low_res = torch.argwhere(low_res_masks[0][0] > 0.0) # 1, 1, 256, 256
                if len(non_zero_indices_low_res) != 0:
                    y_min_low_res, x_min_low_res = non_zero_indices_low_res.min(dim=0).values
                    y_max_low_res, x_max_low_res = non_zero_indices_low_res.max(dim=0).values
                    final_res_bbox_low_res = [
                        round(x_min_low_res.item()/4), 
                        round(y_min_low_res.item()/4), 
                        round(x_max_low_res.item()/4), 
                        round(y_max_low_res.item()/4)
                    ]
                    if final_res_bbox_low_res[0] != final_res_bbox_low_res[2] and final_res_bbox_low_res[1] != final_res_bbox_low_res[3]:
                        # backbone_features : 1, 256, 64, 64
                        local_feature = backbone_features[0, :, final_res_bbox_low_res[1]:final_res_bbox_low_res[3], final_res_bbox_low_res[0]:final_res_bbox_low_res[2]]
                        pooled_local_feature = F.adaptive_avg_pool2d(local_feature, (1, 1)).squeeze()
                        average_history_feature = torch.mean(torch.stack(self.feature_bank_for_error_detect_recovery), dim=0)
                        feature_similarity = cos(pooled_local_feature, average_history_feature)
                        if feature_similarity < self.error_detect_adjacent_frames_feature_similarity_threshold:
                            is_Feature_error = True
                            # print(f"⚠️Found possible error, Feature similarity:{feature_similarity}")
            
        if self.error_detect_strategy == "or":
            return is_IoU_error or is_Shape_acc_error or is_Size_acc_error or is_Feature_error
        elif self.error_detect_strategy == "and":
            return is_IoU_error and is_Shape_acc_error and is_Size_acc_error and is_Feature_error
        else:
            raise ValueError(f"Invalid error detect strategy: {self.error_detect_strategy}")
    
    def _judge_error_recovery(
        self,
        high_res_multibboxes,
        low_res_multibboxes,
        backbone_features,
        device,
        original_best_iou_inds,
        low_res_multimasks=None,
        original_ious=None,
    ):  
        if not self.is_current_segmentation_likely_to_be_error or not self.error_detection_mode:
            print("⚠️Error recovery activated, but no error detected.")
            return original_best_iou_inds
        if len(self.bbox_bank_for_error_detect_recovery) == 0 or len(self.feature_bank_for_error_detect_recovery) == 0:
            self.is_current_segmentation_likely_to_be_error = False
            return original_best_iou_inds
        # Average all boxes in self.bbox_bank_for_error_detect_recovery as the new bbox
        average_bbox = np.mean(self.bbox_bank_for_error_detect_recovery, axis=0)
        average_bbox = self.convert_xyxy_to_current_state(average_bbox)
        average_history_feature = torch.mean(torch.stack(self.feature_bank_for_error_detect_recovery), dim=0)
        
        error_recovery_size_similarity_scores = torch.tensor(compute_size_score(average_bbox, high_res_multibboxes, state_type=self.state_type), device=device)
        error_recovery_shape_similarity_scores = torch.tensor(compute_shape_score(average_bbox, high_res_multibboxes, state_type=self.state_type), device=device)
        error_recovery_feature_similarity_scores = self._compute_feature_similarity_score(average_history_feature, backbone_features, low_res_multibboxes, low_res_multimasks)
        
        good_inds = []
        for i in range(len(high_res_multibboxes)):
            error_recovery_size_similarity_score = error_recovery_size_similarity_scores[i]
            error_recovery_shape_similarity_score = error_recovery_shape_similarity_scores[i]
            error_recovery_feature_similarity_score = error_recovery_feature_similarity_scores[i]
            if error_recovery_size_similarity_score >= self.error_recovery_size_similarity_threshold \
                and error_recovery_shape_similarity_score >= self.error_recovery_shape_similarity_threshold \
                    and error_recovery_feature_similarity_score >= self.error_recovery_feature_similarity_threshold:
                good_inds.append(i)
                self.is_current_segmentation_likely_to_be_error = False
                # Clear all memory banks
                self.mask_bank_for_error_detect_recovery = []
                self.bbox_bank_for_error_detect_recovery = []
                self.feature_bank_for_error_detect_recovery = []
        if len(good_inds) > 0:
            if self.recover_at_best_iou:
                good_ious = original_ious.squeeze().float().cpu().numpy()[good_inds]
                best_iou_ind = good_ious.argmax()
                # print(f"🟢Exit error period")
                return good_inds[best_iou_ind]
            else:
                # print(f"🟢Exit error period")
                return good_inds[0]
        else:
            return original_best_iou_inds
    
    
    def _forward_sam_heads(
        self,
        backbone_features, # (1, 256, 64, 64)
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.min_obj_score_logits

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        kf_ious = None
        mp_ious = None
        if multimask_output and self.samurai_mode:
            if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                self.kf_mean, self.kf_covariance = self.kf.initiate(self.convert_xyxy_to_current_state(high_res_bbox))
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
                self.stable_frames += 1
            elif self.stable_frames < self.stable_frames_threshold:
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                if ious[0][best_iou_inds] > self.stable_ious_threshold:
                    self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.convert_xyxy_to_current_state(high_res_bbox))
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
            else:
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                high_res_multibboxes = []
                low_res_multibboxes = []
                batch_inds = torch.arange(B, device=device)
                for i in range(ious.shape[1]):
                    non_zero_indices = torch.argwhere(high_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_multibboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
                for i in range(ious.shape[1]):
                    non_zero_indices_low = torch.argwhere(low_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices_low) == 0:
                        low_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min_low, x_min_low = non_zero_indices_low.min(dim=0).values
                        y_max_low, x_max_low = non_zero_indices_low.max(dim=0).values
                        low_res_multibboxes.append([
                            round(x_min_low.item()/4), 
                            round(y_min_low.item()/4), 
                            round(x_max_low.item()/4), 
                            round(y_max_low.item()/4)
                        ])
                # compute the IoU between the predicted bbox and the high_res_multibboxes
                kf_ious = torch.tensor(compute_iou(self.kf_mean[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                mp_shape_score = torch.tensor(compute_shape_score(self.kf_mean[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                mp_size_score = torch.tensor(compute_size_score(self.kf_mean[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                mp_center_distance_score = torch.tensor(compute_center_distance_score(self.kf_mean[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                # weighted iou
                weighted_ious = self.kf_score_weight * kf_ious + self.sam2_iou_score_weight * ious
                weighted_score = weighted_ious \
                               + self.mp_shape_score_weight * mp_shape_score \
                               + self.mp_size_score_weight * mp_size_score \
                               + self.mp_center_distance_score_weight * mp_center_distance_score
                best_iou_inds = torch.argmax(weighted_score, dim=-1)
                batch_inds = torch.arange(B, device=device)
                
                if self.error_detection_mode and self.is_current_segmentation_likely_to_be_error:
                    best_iou_inds = self._judge_error_recovery(high_res_multibboxes, low_res_multibboxes, backbone_features, device, best_iou_inds, low_res_multimasks[0], weighted_score)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1

                if ious[0][best_iou_inds] < self.stable_ious_threshold:
                    self.stable_frames = 0
                else:
                    self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.convert_xyxy_to_current_state(high_res_multibboxes[best_iou_inds]))
        elif multimask_output and self.samosa_mode:
            if self.motion_predictor.history_bank_length() == 0:
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                # If there is no valid mask, initialize bbox as all zeros
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                # Initialize the MP model's history_bank
                initial_state = self.motion_predictor.get_full_state(self.convert_xyxy_to_current_state(high_res_bbox))
                self.motion_predictor.predict(initial_state)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
                self.stable_frames += 1
                mp_ious = None
            elif self.motion_predictor.history_bank_length() < self.motion_predictor.history_size:
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                # If there is no valid mask, use an all-zero bbox when updating the MP model's history_bank
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                # If stable enough, update the MP model's history_bank; otherwise, clear history_bank
                if ious[0][best_iou_inds] > self.mp_stable_ious_threshold:
                    full_state = self.motion_predictor.get_full_state(self.convert_xyxy_to_current_state(high_res_bbox))
                    self.motion_predictor.predict(full_state)
                    self.stable_frames += 1
                else:
                    self.motion_predictor.clear_banks()
                    self.stable_frames = 0
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
                mp_ious = None
            # Already stable, so MP prediction and IoU computation can be performed
            else:
                predicted_state = self.motion_predictor.predict()
                high_res_multibboxes = []
                low_res_multibboxes = []
                batch_inds = torch.arange(B, device=device)
                for i in range(ious.shape[1]):
                    non_zero_indices = torch.argwhere(high_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_multibboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
                for i in range(ious.shape[1]):
                    non_zero_indices_low = torch.argwhere(low_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices_low) == 0:
                        low_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min_low, x_min_low = non_zero_indices_low.min(dim=0).values
                        y_max_low, x_max_low = non_zero_indices_low.max(dim=0).values
                        low_res_multibboxes.append([
                            round(x_min_low.item()/4), 
                            round(y_min_low.item()/4), 
                            round(x_max_low.item()/4), 
                            round(y_max_low.item()/4)
                        ])
                # compute the IoU between the predicted bbox and the high_res_multibboxes
                mp_ious = torch.tensor(compute_iou(predicted_state[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                mp_shape_score = torch.tensor(compute_shape_score(predicted_state[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                mp_size_score = torch.tensor(compute_size_score(predicted_state[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                mp_center_distance_score = torch.tensor(compute_center_distance_score(predicted_state[:4], high_res_multibboxes, state_type=self.state_type), device=device)
                # weighted iou
                weighted_ious = self.mp_iou_score_weight * mp_ious + self.sam2_iou_score_weight * ious
                weighted_score = weighted_ious \
                               + self.mp_shape_score_weight * mp_shape_score \
                               + self.mp_size_score_weight * mp_size_score \
                               + self.mp_center_distance_score_weight * mp_center_distance_score
                best_iou_inds = torch.argmax(weighted_score, dim=-1)
                batch_inds = torch.arange(B, device=device)
                
                if self.error_detection_mode and self.is_current_segmentation_likely_to_be_error:
                    best_iou_inds = self._judge_error_recovery(high_res_multibboxes, low_res_multibboxes, backbone_features, device, best_iou_inds, low_res_multimasks[0], weighted_score)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1

                if ious[0][best_iou_inds] < self.mp_stable_ious_threshold:
                    self.stable_frames = 0
                    self.motion_predictor.clear_banks()
                else:
                    full_state = self.motion_predictor.get_full_state(self.convert_xyxy_to_current_state(high_res_multibboxes[best_iou_inds]))
                    self.motion_predictor.predict(full_state)
                    
        elif multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            
            if self.error_detection_mode and self.is_current_segmentation_likely_to_be_error:
                best_iou_inds = self._judge_error_recovery(high_res_multibboxes, low_res_multibboxes, backbone_features, device, best_iou_inds, low_res_multimasks[0], ious)
                      
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            best_iou_inds = 0
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        frame_obj_feature = None
        # Update the reselect box and feature memory banks, and get the frame feature for memory selection
        if self.memory_selection_init_feat_similarity_weight > 0.0 or self.memory_bank_init_feat_similarity_threshold > 0.0 \
                or self.memory_selection_last_feat_similarity_weight > 0.0 or self.memory_bank_last_feat_similarity_threshold > 0.0:
            
            # Crop local features
            if self.crop_feature_by_mask:
                # Feature extraction based on the mask
                mask = low_res_masks[0][0]  # [256, 256]
                # Binarize the mask: values greater than 0 become 1, values less than or equal to 0 become 0
                mask = (mask > 0).float()
                if mask.numel() > 0 and mask.sum() > 0:
                    # Resize the mask to the backbone_features size
                    mask_resized = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(backbone_features.size(2), backbone_features.size(3)),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    # Perform weighted average pooling based on the mask
                    masked_features = backbone_features[0] * mask_resized.unsqueeze(0)  # [256, 64, 64]
                    mask_sum = mask_resized.sum()
                    
                    if mask_sum > 0:
                        pooled_local_feature = masked_features.sum(dim=(1, 2)) / mask_sum  # [256]
                    else:
                        pooled_local_feature = torch.zeros(backbone_features.size(1), device=backbone_features.device)
                else:
                    pooled_local_feature = torch.zeros(backbone_features.size(1), device=backbone_features.device)
            else:
                # Feature extraction based on bounding boxes (original implementation)
                non_zero_indices_low_res = torch.argwhere(low_res_masks[0][0] > 0.0) # 1, 1, 256, 256
                if len(non_zero_indices_low_res) != 0:
                    y_min_low_res, x_min_low_res = non_zero_indices_low_res.min(dim=0).values
                    y_max_low_res, x_max_low_res = non_zero_indices_low_res.max(dim=0).values
                    final_res_bbox_low_res = [
                        round(x_min_low_res.item()/4), 
                        round(y_min_low_res.item()/4), 
                        round(x_max_low_res.item()/4), 
                        round(y_max_low_res.item()/4)
                    ]
                    if final_res_bbox_low_res[0] != final_res_bbox_low_res[2] and final_res_bbox_low_res[1] != final_res_bbox_low_res[3]:
                        
                        # backbone_features : 1, 256, 64, 64
                        local_feature = backbone_features[0, :, final_res_bbox_low_res[1]:final_res_bbox_low_res[3], final_res_bbox_low_res[0]:final_res_bbox_low_res[2]]
                        pooled_local_feature = F.adaptive_avg_pool2d(local_feature, (1, 1)).squeeze()
                    else:
                        pooled_local_feature = torch.zeros(backbone_features.size(1), device=backbone_features.device)
                else:
                    pooled_local_feature = torch.zeros(backbone_features.size(1), device=backbone_features.device)
            
            if self.memory_selection_init_feat_similarity_weight > 0.0 or self.memory_bank_init_feat_similarity_threshold > 0.0 \
                or self.memory_selection_last_feat_similarity_weight > 0.0 or self.memory_bank_last_feat_similarity_threshold > 0.0:
                frame_obj_feature = pooled_local_feature
        
        # Error detection
        if self.error_detection_mode and not self.is_current_segmentation_likely_to_be_error:
            # Compute the bbox from high_res_masks
            non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                current_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(dim=0).values
                y_max, x_max = non_zero_indices.max(dim=0).values
                current_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
            
            # Run error detection; this step only performs detection and does not manage error detection memory
            self.is_current_segmentation_likely_to_be_error = self._detect_error(high_res_masks, current_bbox, backbone_features, low_res_masks)

            # Update memory banks for error detection and exit only when the current frame is considered error-free
            if not self.is_current_segmentation_likely_to_be_error:

                # Update the mask memory bank
                self.mask_bank_for_error_detect_recovery.append(high_res_masks.detach().cpu())
                if len(self.mask_bank_for_error_detect_recovery) > self.error_detect_recovery_history_length:
                    self.mask_bank_for_error_detect_recovery.pop(0)
                # Update the bbox memory bank
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    final_res_bbox = [0, 0, 0, 0]
                    if not self.update_only_exist_box_for_error_recovery:
                        self.bbox_bank_for_error_detect_recovery.append(final_res_bbox)
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    final_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    self.bbox_bank_for_error_detect_recovery.append(final_res_bbox)
                if len(self.bbox_bank_for_error_detect_recovery) > self.error_detect_recovery_history_length:
                    self.bbox_bank_for_error_detect_recovery.pop(0)
                # Crop local features
                if self.crop_feature_by_mask:
                    # Feature extraction based on the mask
                    mask = low_res_masks[0][0]  # [256, 256]
                    # Binarize the mask: values greater than 0 become 1, values less than or equal to 0 become 0
                    mask = (mask > 0).float()
                    if mask.numel() > 0 and mask.sum() > 0:
                        # Resize the mask to the backbone_features size
                        mask_resized = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(backbone_features.size(2), backbone_features.size(3)),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                        
                        # Perform weighted average pooling based on the mask
                        masked_features = backbone_features[0] * mask_resized.unsqueeze(0)  # [C, H, W]
                        mask_sum = mask_resized.sum()
                        
                        if mask_sum > 0:
                            pooled_local_feature = masked_features.sum(dim=(1, 2)) / mask_sum  # [C]
                        else:
                            pooled_local_feature = torch.zeros(backbone_features.size(1), device=backbone_features.device)
                        
                        self.feature_bank_for_error_detect_recovery.append(pooled_local_feature)
                        
                        if len(self.feature_bank_for_error_detect_recovery) > self.error_detect_recovery_history_length:
                            self.feature_bank_for_error_detect_recovery.pop(0)
                else:
                    # Feature extraction based on bounding boxes (original implementation)
                    non_zero_indices_low_res = torch.argwhere(low_res_masks[0][0] > 0.0) # 1, 1, 256, 256
                    if len(non_zero_indices_low_res) != 0:
                        y_min_low_res, x_min_low_res = non_zero_indices_low_res.min(dim=0).values
                        y_max_low_res, x_max_low_res = non_zero_indices_low_res.max(dim=0).values
                        final_res_bbox_low_res = [
                            round(x_min_low_res.item()/4), 
                            round(y_min_low_res.item()/4), 
                            round(x_max_low_res.item()/4), 
                            round(y_max_low_res.item()/4)
                        ]
                        if final_res_bbox_low_res[0] != final_res_bbox_low_res[2] and final_res_bbox_low_res[1] != final_res_bbox_low_res[3]:
                            
                            # backbone_features : 1, 256, 64, 64
                            local_feature = backbone_features[0, :, final_res_bbox_low_res[1]:final_res_bbox_low_res[3], final_res_bbox_low_res[0]:final_res_bbox_low_res[2]]
                            pooled_local_feature = F.adaptive_avg_pool2d(local_feature, (1, 1)).squeeze()
                            self.feature_bank_for_error_detect_recovery.append(pooled_local_feature)
                            
                            if len(self.feature_bank_for_error_detect_recovery) > self.error_detect_recovery_history_length:
                                self.feature_bank_for_error_detect_recovery.pop(0)
            
        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        if self.samurai_mode:
            motion_ious = kf_ious
        elif self.samosa_mode:
            motion_ious = mp_ious
        else:
            motion_ious = None

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            ious[0][best_iou_inds],
            motion_ious[best_iou_inds] if motion_ious is not None else None,
            frame_obj_feature,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _, _, _, frame_obj_feature = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            frame_obj_feature,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            selected_cond_frames = selected_cond_outputs.keys()
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.samurai_mode or self.samosa_mode:
                if self.memory_selection_strategy == "backward":
                    valid_indices = [] 
                    chosen_error_memory_num = 0
                    init_frame_obj_feature = t_pos_and_prevs[0][1]["frame_obj_feature"] if "frame_obj_feature" in t_pos_and_prevs[0][1] else None
                    last_frame_idx = frame_idx - 1
                    last_frame_obj_feature = None if last_frame_idx not in output_dict["non_cond_frame_outputs"] \
                        else (output_dict["non_cond_frame_outputs"][last_frame_idx]["frame_obj_feature"] if "frame_obj_feature" in output_dict["non_cond_frame_outputs"][last_frame_idx] else None)
                    if track_in_reverse:
                        raise NotImplementedError("`track_in_reverse` not supported for SAMURAI/Markov mode")
                    if frame_idx > 1:  # Ensure we have previous frames to evaluate
                    # if frame_idx > 1 and "best_iou_score" in output_dict["non_cond_frame_outputs"][frame_idx - 1]:  # Ensure we have previous frames to evaluate
                        # for i in range(frame_idx - 1, 1, -1):  # Iterate backwards through previous frames
                        for i in range(frame_idx - 1, len(selected_cond_outputs), -1):  # Iterate backwards through previous frames
                            if i not in output_dict["non_cond_frame_outputs"]:
                                continue
                            
                            # Decide whether to discard memories from error periods: when the current state has no error, ignore memories marked as potential errors; when the current pool has reached the error-memory limit, ignore memories marked as potential errors
                            ready_to_add_error_memory = False
                            if self.error_detection_mode and (not self.is_current_segmentation_likely_to_be_error) and  "likely_to_be_error" in output_dict["non_cond_frame_outputs"][i]:
                                if output_dict["non_cond_frame_outputs"][i]["likely_to_be_error"]:
                                    if self.ignore_error_memories: 
                                        continue
                                    elif chosen_error_memory_num >= self.restrict_max_error_memory_num_in_pool:
                                        continue
                                    else:
                                        ready_to_add_error_memory = True

                            # Read or compute scores
                            iou_score = output_dict["non_cond_frame_outputs"][i]["best_iou_score"] if "best_iou_score" in output_dict["non_cond_frame_outputs"][i] else None
                            obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"] if "object_score_logits" in output_dict["non_cond_frame_outputs"][i] else None
                            if self.apply_sigmoid_to_obj_score_logits:
                                obj_score = obj_score.sigmoid()
                            kf_score = output_dict["non_cond_frame_outputs"][i]["kf_score"] if "kf_score" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available
                            cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
                            frame_obj_feature = output_dict["non_cond_frame_outputs"][i]["frame_obj_feature"]
                            init_feat_similarity_score = cosine_similarity(frame_obj_feature, init_frame_obj_feature) if (init_frame_obj_feature is not None and frame_obj_feature is not None) else None
                            last_feat_similarity_score = cosine_similarity(frame_obj_feature, last_frame_obj_feature) if (last_frame_obj_feature is not None and frame_obj_feature is not None) else None
                            # Check if the scores meet the criteria for being a valid index
                            if (iou_score is None or iou_score.item() > self.memory_bank_iou_threshold) and \
                            (obj_score is None or obj_score.item() > self.memory_bank_obj_score_threshold) and \
                            (kf_score is None or kf_score.item() > self.memory_bank_kf_score_threshold) and \
                            (init_feat_similarity_score is None or init_feat_similarity_score.item() > self.memory_bank_init_feat_similarity_threshold) and \
                            (last_feat_similarity_score is None or last_feat_similarity_score.item() > self.memory_bank_last_feat_similarity_threshold):
                                valid_indices.insert(0, i)  
                                if ready_to_add_error_memory:
                                    chosen_error_memory_num += 1
                            # Check the number of valid indices
                            if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:  
                                break
                    # if frame_idx - 1 not in valid_indices: 
                    #     valid_indices.append(frame_idx - 1)
                    for i in range(self.short_mem_length, 0, -1):  # Iterate over the number of mask memories
                        if (frame_idx - i) not in valid_indices and (frame_idx - i) not in selected_cond_frames:
                            valid_indices.append(frame_idx - i)
                    for t_pos in range(len(selected_cond_outputs), self.num_maskmem):  # Iterate over the number of mask memories
                        idx = t_pos - self.num_maskmem  # Calculate the index for valid indices
                        if idx < -len(valid_indices):  # Skip if index is out of bounds
                            continue
                        out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)  # Get output for the valid index
                        if out is None:  # If not found, check unselected outputs
                            out = unselected_cond_outputs.get(valid_indices[idx], None)
                        t_pos_and_prevs.append((t_pos, out))  # Append the temporal position and output to the list
                elif self.memory_selection_strategy == "topk":
                    
                    valid_indices = [] 
                    weighted_scores = []
                    chosen_error_memory_num = 0
                    init_frame_obj_feature = t_pos_and_prevs[0][1]["frame_obj_feature"] if "frame_obj_feature" in t_pos_and_prevs[0][1] else None
                    last_frame_idx = frame_idx - 1
                    last_frame_obj_feature = None if last_frame_idx not in output_dict["non_cond_frame_outputs"] \
                        else (output_dict["non_cond_frame_outputs"][last_frame_idx]["frame_obj_feature"] if "frame_obj_feature" in output_dict["non_cond_frame_outputs"][last_frame_idx] else None)
                    if track_in_reverse:
                        raise NotImplementedError("`track_in_reverse` not supported for SAMURAI/Markov mode")
                    if frame_idx > 1:  # Ensure we have previous frames to evaluate
                    # if frame_idx > 1 and "best_iou_score" in output_dict["non_cond_frame_outputs"][frame_idx - 1]:  # Ensure we have previous frames to evaluate
                        # for i in range(frame_idx - 1, 1, -1):  # Iterate backwards through previous frames
                        for i in range(frame_idx - 1, len(selected_cond_outputs), -1):  # Iterate backwards through previous frames
                            if i not in output_dict["non_cond_frame_outputs"]:
                                continue
                            
                            # Decide whether to discard memories from error periods: when the current state has no error, ignore memories marked as potential errors; when the current pool has reached the error-memory limit, ignore memories marked as potential errors
                            ready_to_add_error_memory = False
                            if self.error_detection_mode and (not self.is_current_segmentation_likely_to_be_error) and  "likely_to_be_error" in output_dict["non_cond_frame_outputs"][i]:
                                if output_dict["non_cond_frame_outputs"][i]["likely_to_be_error"]:
                                    if self.ignore_error_memories: 
                                        continue
                                    elif chosen_error_memory_num >= self.restrict_max_error_memory_num_in_pool:
                                        continue
                                    else:
                                        ready_to_add_error_memory = True
                                    
                            # Read or compute scores
                            iou_score = output_dict["non_cond_frame_outputs"][i]["best_iou_score"] if "best_iou_score" in output_dict["non_cond_frame_outputs"][i] else None
                            obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"] if "object_score_logits" in output_dict["non_cond_frame_outputs"][i] else None
                            if self.apply_sigmoid_to_obj_score_logits:
                                obj_score = obj_score.sigmoid()
                            kf_score = output_dict["non_cond_frame_outputs"][i]["kf_score"] if "kf_score" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available
                            cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
                            frame_obj_feature = output_dict["non_cond_frame_outputs"][i]["frame_obj_feature"]
                            init_feat_similarity_score = cosine_similarity(frame_obj_feature, init_frame_obj_feature) if (init_frame_obj_feature is not None and frame_obj_feature is not None) else None
                            last_feat_similarity_score = cosine_similarity(frame_obj_feature, last_frame_obj_feature) if (last_frame_obj_feature is not None and frame_obj_feature is not None) else None
                            # Check if the scores meet the criteria for being a valid index
                            if (iou_score is None or iou_score.item() > self.memory_bank_iou_threshold) and \
                            (obj_score is None or obj_score.item() > self.memory_bank_obj_score_threshold) and \
                            (kf_score is None or kf_score.item() > self.memory_bank_kf_score_threshold) and \
                            (init_feat_similarity_score is None or init_feat_similarity_score.item() > self.memory_bank_init_feat_similarity_threshold) and \
                            (last_feat_similarity_score is None or last_feat_similarity_score.item() > self.memory_bank_last_feat_similarity_threshold):
                                valid_indices.insert(0, i)  
                                # Compute the weighted score
                                iou_score = 0.0 if iou_score is None else iou_score.item()
                                obj_score = 0.0 if obj_score is None else obj_score.item()
                                kf_score = 0.0 if kf_score is None else kf_score.item()
                                init_feat_similarity_score = 0.0 if init_feat_similarity_score is None else init_feat_similarity_score.item()
                                last_feat_similarity_score = 0.0 if last_feat_similarity_score is None else last_feat_similarity_score.item()
                                weighted_scores.insert(0, \
                                    iou_score * self.memory_selection_sam2iou_weight \
                                        + obj_score * self.memory_selection_obj_score_weight \
                                            + kf_score * self.memory_selection_kf_score_weight \
                                                + init_feat_similarity_score * self.memory_selection_init_feat_similarity_weight \
                                                    + last_feat_similarity_score * self.memory_selection_last_feat_similarity_weight)
                                if ready_to_add_error_memory:
                                    chosen_error_memory_num += 1
                            # Check the number of valid indices
                            if len(valid_indices) >= self.memory_selection_range - 1:  
                                break

                        # ========================================
                        # Select good candidates
                        # ========================================
                        num_maskmem = self.num_maskmem - len(t_pos_and_prevs)  # already has 1 cond frame (frame #0, with gt bbox as input)
                        if len(valid_indices) > num_maskmem:
                            ind_last = valid_indices[-1]
                            weighted_scores = torch.tensor(weighted_scores)[:-1]  # always keep the last frame
                            
                            vals, inds = weighted_scores.topk(num_maskmem - 1, dim=0, largest=True, sorted=True)  # 0, num_maskmem - 1
                            inds = inds.squeeze(0)  # num_maskmem - 1
                            inds = inds.sort(descending=False, dim=0)[0]  # sort indices
                            
                            valid_indices = torch.tensor(valid_indices)
                            valid_indices = valid_indices[inds]  # select most similar frames
                            
                            valid_indices = valid_indices.detach().cpu().numpy().tolist()
                            valid_indices.append(ind_last)
                                
                    start_idx = len(t_pos_and_prevs)
                    end_idx = self.num_maskmem
                    for t_pos in range(start_idx, end_idx):  # Iterate over the number of mask memories
                        idx = t_pos - self.num_maskmem  # Calculate the index for valid indices
                        if idx < -len(valid_indices):  # Skip if index is out of bounds
                            continue
                        out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)  # Get output for the valid index
                        if out is None:  # If not found, check unselected outputs
                            out = unselected_cond_outputs.get(valid_indices[idx], None)
                        t_pos_and_prevs.append((t_pos, out))  # Append the temporal position and output to the list
                else:
                    raise ValueError(f"Invalid memory selection strategy: {self.memory_selection_strategy}")
            else:
                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            # first find the nearest frame among every r-th frames before this frame
                            # for r=1, this would be (frame_idx - 2)
                            prev_frame_idx = ((frame_idx - 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if out is None:
                        # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                        # frames, we still attend to it as if it's a non-conditioning frame.
                        out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        if len(sam_outputs) == 8:
            (
                low_res_masks,
                high_res_masks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                frame_obj_feature,
            )= sam_outputs
        else:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                best_iou_score,
                kf_ious,
                frame_obj_feature,
            ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if len(sam_outputs) != 8:
            current_out["best_iou_score"] = best_iou_score
            current_out["kf_ious"] = kf_ious
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        current_out["likely_to_be_error"] = self.is_current_segmentation_likely_to_be_error
        current_out["frame_obj_feature"] = frame_obj_feature

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
