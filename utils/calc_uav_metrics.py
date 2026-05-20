try:
    from utils.metrics import rect_iou, center_error, eval_acc, calc_curves
except:
    from metrics import rect_iou, center_error, eval_acc, calc_curves
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm

def _calc_metrics(boxes, anno):
    ious = rect_iou(boxes, anno) 
    center_errors = center_error(boxes, anno)
    return ious, center_errors

def calc_full_uav_metrics(json_root, metric_save_dir, dataset_name=None, label='Current Method', nbins_iou=21, nbins_ce=51):
    """
    Plot success and precision curves for results of a single method.
    json_root: str, path to the directory containing json files
    plt_save_root: str, path to the directory to save the plots
    label: str, name of the method
    mode: str, (average | overall) how to calculate the result of all videos
        average: calculate the curve for each video and take the average
        overall: concatenate all videos and calculate the curve for the whole dataset
    nbins_iou: int, number of bins for overlap curve
    nbins_ce: int, number of bins for location error curve
    
    """
    succ_curves = []
    prec_curves = []
    clip_accs = []
    clip_accs_dict = {}
    clip_prec_dict = {}
    # for json_file in tqdm(os.listdir(json_root)):
    for json_file in sorted(os.listdir(json_root)):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(json_root, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'gt_rect' not in data or'res' not in data:
            raise ValueError('gt_rect or res not in json file')
        for i, elem in enumerate(data['gt_rect']):
            if elem == [] or elem == [-100, -100, -100, -100]:
                data['gt_rect'][i] = [0, 0, 0, 0]
        for i, elem in enumerate(data['res']):
            if elem == [] or elem == [-100, -100, -100, -100]:
                data['res'][i] = [0, 0, 0, 0]
        gt_bb = data['gt_rect']
        result_bb = data['res']
        exist = data['exist']
        assert len(gt_bb) == len(exist), f"Length of gt_bb is not equal to length of exist: {len(gt_bb)} != {len(exist)} in {json_file}"
        if len(result_bb) > len(gt_bb):
            result_bb = result_bb[:len(gt_bb)]
        if len(result_bb) < len(gt_bb):
            raise ValueError(f"Length of result_bb is less than length of gt_bb: {len(result_bb)} < {len(gt_bb)}")
        ious, center_errors = _calc_metrics(np.array(result_bb), np.array(gt_bb))
        clip_acc = eval_acc(result_bb, gt_bb, exist)
        clip_accs.append(clip_acc)
        one_succ_curve, one_prec_curve = calc_curves(ious, center_errors, nbins_iou, nbins_ce)
        succ_curves.append(one_succ_curve)
        prec_curves.append(one_prec_curve)
        clip_prec_dict[json_file.split('.')[0]] = one_prec_curve[20]
        clip_accs_dict[json_file.split('.')[0]] = clip_acc
    
    succ_curve = np.mean(succ_curves, axis=0)
    prec_curve = np.mean(prec_curves, axis=0)
    succ_score, prec_score, succ_rate = np.mean(succ_curve)*100, prec_curve[20]*100, succ_curve[nbins_iou // 2]*100
    acc = np.mean(clip_accs) * 100
    
    print(f"{len(clip_accs)} Videos, Acc: {acc:.2f}%, prec(P): {prec_score:.2f}%, succ(AUC): {succ_score:.2f}%")
    
    with open(os.path.join(metric_save_dir, 'Main_metircs.txt'), 'w') as f:
        f.write(f"Result Path: {json_root}\n")
        f.write(f"Number of Videos: {len(clip_accs)}\n")
        f.write(f"Acc: {acc:.2f}%\n")
        f.write(f"prec(P): {prec_score:.2f}%\n")
        f.write(f"succ(AUC): {succ_score:.2f}%\n")

    with open(os.path.join(metric_save_dir, 'Accuracy_per_video.txt'), 'w') as f:
        for key, value in clip_accs_dict.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Overall Accuracy: {acc}\n")
    with open(os.path.join(metric_save_dir, 'Precision_per_video.txt'), 'w') as f:
        for key, value in clip_prec_dict.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Overall Precision: {prec_score}\n")

    # Plot success curve
    x_overlap = np.linspace(0, 1, nbins_iou)  # x-axis ranges from 0 to 1
    plt.plot(x_overlap, succ_curve)
    plt.xlabel('Overlap Threshold')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate Curve for {dataset_name}')
    plt.legend([f'{label} [{succ_score:.2f}]'])
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid
    plt.xticks(np.linspace(0, 1, 11))  # Adjust x-ticks
    plt.yticks(np.linspace(0, 1, 11))  # Adjust y-ticks
    plt.savefig(os.path.join(metric_save_dir, 'overlap_success.png'))
    plt.clf()

    # Plot precision curve
    x_error = np.arange(0, nbins_ce)  # x-axis is the error threshold
    plt.plot(x_error, prec_curve)
    plt.xlabel('Location Error Threshold')
    plt.ylabel('Precision')
    plt.title(f'Precision Curve for {dataset_name}')
    plt.legend([f'{label} [{prec_score:.2f}]'])
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid
    plt.xticks(np.linspace(0, nbins_ce-1, 11))  # Adjust x-ticks
    plt.yticks(np.linspace(0, 1, 11))  # Adjust y-ticks
    plt.savefig(os.path.join(metric_save_dir, 'error_prec.png'))
    plt.clf()

def calc_and_save_metrics(res_root):
    folders = [
        'Anti-UAV300/test',
        'Anti-UAV300/test_infrared',
        'Anti-UAV300/test-dev',
        'Anti-UAV300/test-dev_IR',
        'Anti-UAV410/test',
        'Anti-UAV410/test-dev',
        'Anti-UAV600/val',
        'DUT',
        'AntiUAV300/test',
        'AntiUAV300/test_infrared',
        'AntiUAV300/test-dev',
        'AntiUAV300/test-dev_IR',
        'AntiUAV410/test',
        'AntiUAV410/test-dev',
        'AntiUAV600/val',
    ]
    for folder in folders:
        dataset_res_root = f'{res_root}/{folder}'
        metric_save_dir = os.path.join(dataset_res_root, 'metrics')
        dataset_name = dataset_res_root.split('/')[-2] + '_' + dataset_res_root.split('/')[-1]
        if not os.path.exists(dataset_res_root):
            continue  
        os.makedirs(metric_save_dir, exist_ok=True)
        print(f'-------------------{folder}-------------------')
        calc_full_uav_metrics(dataset_res_root, metric_save_dir, dataset_name=dataset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", default="output/samosa/", help="path to test video path")
    args = parser.parse_args()

    calc_and_save_metrics(args.res_path)
