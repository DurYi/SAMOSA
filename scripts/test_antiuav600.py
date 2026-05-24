import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import tqdm
import json
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
from utils.metrics import eval_acc
from utils.calc_uav_metrics import calc_and_save_metrics
from utils.utils import determine_model_cfg, load_gt_label
color = [(0, 255, 0)]
                        
def load_json_point(gt_path):
    with open(gt_path, 'r') as f:
        data = json.load(f)  # Read JSON data
    prompts = {}
    points = {}
    flag = False 
    index = len(data['exist'])
    for index in range(len(data['exist'])):
        if data['exist'][index] == 1:
            break
    first_frame_label = data['gt_rect'][index] 
    if index == len(data['exist']) - 1:
        flag = True
        return prompts, points, flag
    x, y, w, h = first_frame_label  # Get x, y, w, h
    # Convert top-left coordinates and width/height to center-point coordinates
    x_center, y_center = (x + w / 2), (y + h / 2)
    x_center, y_center, w, h = int(x_center), int(y_center), int(w), int(h)
    prompts[0] = ((x, y, x + w, y + h), 0)  # Given box top-left and bottom-right coordinates
    points[0] = [[x_center, y_center]]  # Save only the center-point coordinates
    return prompts, points, flag, index

def main(args):
    print(args)
    model_cfg = determine_model_cfg(args.model_path, args.mode)
    predictor = build_sam2_video_predictor(model_cfg, ckpt_path=args.model_path, mp_ckpt_path=args.mp_model_path, device="cuda:0")
    test_video_names = os.listdir(args.test_path)
    test_video_names.sort()
    method_name = 'samosa' if 'samosa' in args.mode else args.mode
    out_folder = f"{args.out_root}/{method_name}{args.output_suffix}/Anti-UAV600/val"
    
    for i, test_video_name in enumerate(test_video_names): 
        if osp.exists(f"{out_folder}/{test_video_name}.json"):
            print(f"Skipped {test_video_name} as it has been processed.")
            continue
           
        video_folder = osp.join(args.test_path, test_video_name)
        frames_path = video_folder
        print(f"({i+1}/{len(test_video_names)}):{frames_path}")
        json_path = osp.join(video_folder, 'IR_label.json')
        gt_traj = load_gt_label(json_path)
        prompts, points, flag, index = load_json_point(json_path)
        if osp.isdir(frames_path):
            frames = sorted([osp.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith(".jpg")])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(frames_path)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        if args.save_to_video:
            out = cv2.VideoWriter(f"{out_folder}/{test_video_name}.mp4", fourcc, 30, (width, height))

        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(frames_path, offload_video_to_cpu=True)
            bbox, track_label = prompts[0]
            point_prompt = np.array(points[0], np.int32)
            labels = np.array([1], np.int32)
            if args.use_point_prompt and not args.use_bbox_prompt:
                _, _, masks = predictor.add_new_points_or_box(state, points=point_prompt, labels = labels, frame_idx=index, obj_id=0)
            elif args.use_bbox_prompt and not args.use_point_prompt:
                _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=index, obj_id=0)
            elif args.use_bbox_prompt and args.use_point_prompt:
                _, _, masks = predictor.add_new_points_or_box(state, points=point_prompt, box=bbox, labels = labels, frame_idx=index, obj_id=0)
            else:
                raise ValueError("Both use_point_prompt and use_bbox_prompt are False!")
            
            pred_traj = [[0,0,0,0] for _ in range(index)]
            bbox_results = [[0,0,0,0] for _ in range(index)]
            mask_results = [None for _ in range(index)]
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                mask_to_vis = {}
                bbox_to_vis = {}

                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bbox_to_vis[obj_id] = bbox
                    pred_traj.append(bbox)
                    mask_to_vis[obj_id] = mask

                if args.save_to_video:
                    img = loaded_frames[frame_idx]
                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                    for obj_id, bbox in bbox_to_vis.items():
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)
                        gt_bbox = gt_traj["gt_rect"][frame_idx]
                        if len(gt_bbox) == 0:
                            gt_bbox = [0, 0, 0, 0]
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 0, 255), 2)
                    out.write(img)
                if args.save_bbox_to_json:
                    bbox_results.append(bbox_to_vis[0])
                if args.save_mask_to_npy:
                    mask_results.append(mask_to_vis[0])

            if args.save_to_video:
                out.release()
            if args.save_bbox_to_json:
                json_data = {
                    "res": bbox_results,
                    "gt_rect": gt_traj["gt_rect"],
                    "exist": gt_traj["exist"]
                }
                with open(f"{out_folder}/{test_video_name}.json", "w") as f:
                    json.dump(json_data, f, indent=4)
            if args.save_mask_to_npy:
                np.save(f"{out_folder}/{test_video_name}.npy", mask_results)
                
        assert len(pred_traj)==len(gt_traj["gt_rect"]), f"The length of pred_traj and gt_traj is not the same for {test_video_name}"  # Ensure the two lists have the same length         
        clip_acc = eval_acc(pred_traj, gt_traj["gt_rect"], gt_traj["exist"])
        print("The accuracy for {} is {}".format(test_video_name, clip_acc))

        # Clear the GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

    calc_and_save_metrics(f"{args.out_root}/{method_name}{args.output_suffix}")
    try:
        del predictor, state
    except:
        print("No video is processed during this run.")
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1", "y"):  # truthy strings
            return True
        if v.lower() in ("no", "false", "f", "0", "n"):  # falsy strings
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="output/", help="path to test video path")
    parser.add_argument("--test_path", default="data/Anti-UAV600/validation/", help="path to test video path")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_large.pt", help="Path to the model checkpoint.")
    
    parser.add_argument("--mp_model_path", default="sam2/checkpoints/mp.pth", help="Path to the markov model checkpoint.")
    parser.add_argument("--mode", default="samosa_b", help="Mode to use (samosa | samurai | sam2.1 | ...)")
    parser.add_argument("--output_suffix", default="", help="Suffix to output folder")
    
    parser.add_argument("--no_ignore", default=True, help="Do not ignore videos without target in the first frame but with target in the rest frames.")
    parser.add_argument("--inference_on_mp4", default=False, help="Inference on mp4 instead of jpg frames.")
    parser.add_argument("--use_point_prompt", type=str2bool, default=True, help="Whether to input point prompt.")
    parser.add_argument("--use_bbox_prompt", type=str2bool, default=True, help="Whether to input bbox prompt.")
    parser.add_argument("--save_bbox_to_json", type=str2bool, default=True, help="Save bbox results to a json.")
    parser.add_argument("--save_mask_to_npy", type=str2bool, default=False, help="Save mask results to a npy.")
    parser.add_argument("--save_to_video", type=str2bool, default=True, help="Save results to a video.")

    args = parser.parse_args()
    main(args)
