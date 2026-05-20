import numpy as np
import json
import os
import os.path as osp

def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                        (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

def determine_model_cfg(model_path, mode):
    if mode in os.listdir('./sam2/sam2/configs'):
        if "large" in model_path and "sam2.1_" in model_path:
            return f"configs/{mode}/sam2.1_hiera_l.yaml"
        elif "large" in model_path and "sam2_" in model_path:
            return f"configs/{mode}/sam2_hiera_l.yaml"
        elif "base_plus" in model_path:
            return f"configs/{mode}/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return f"configs/{mode}/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return f"configs/{mode}/sam2.1_hiera_t.yaml"
        raise ValueError("Unknown model size in path!")
    raise ValueError("Unknown mode!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def load_gt_label(gt_path):
    with open(gt_path, 'r') as f:
        data = json.load(f)  # Read JSON data
    return data
