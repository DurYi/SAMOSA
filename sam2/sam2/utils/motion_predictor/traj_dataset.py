import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import gc

DEFAULT_INIT_WH = (256, 256)
DEFAULT_INIT_POS = (0, 0)
DEFAULT_IMG_WH = (1024, 1024)
  
def xywh_to_xcycwh(bbox):
    """
    Convert a bbox in [x, y, w, h] or [x, y, w, h, dx, dy, dw, dh] format to [xc, yc, w, h] or [xc, yc, w, h, dx, dy, dw, dh] format.
    """
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    xc, yc = x + w / 2, y + h / 2
    if len(bbox) == 4:
        return [xc, yc, w, h]
    elif len(bbox) == 8:
        dx, dy, dw, dh = bbox[4], bbox[5], bbox[6], bbox[7]
        return [xc, yc, w, h, dx, dy, dw, dh]
    else:
        raise ValueError("Unknown bbox format.")

def xcycwh_to_xywh(bbox):
    """
    Convert a bbox in [xc, yc, w, h] or [xc, yc, w, h, dx, dy, dw, dh] format to [x, y, w, h] or [x, y, w, h, dx, dy, dw, dh] format.
    """
    xc, yc, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = xc - w / 2, yc - h / 2
    if len(bbox) == 4:
        return [x, y, w, h]
    elif len(bbox) == 8:
        dxc, dyc, dw, dh = bbox[4], bbox[5], bbox[6], bbox[7]
        dx = dxc - dw / 2
        dy = dyc - dh / 2
        return [x, y, w, h, dx, dy, dw, dh]
    else:
        raise ValueError("Unknown bbox format.")

def xywh_to_x1y1ah(bbox):
    """
    Convert a bbox in [x, y, w, h] or [x, y, w, h, dx, dy, dw, dh] format to [x1, y1, a, h] or [x1, y1, a, h, dx1, dy1, da, dh] format.
    """
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # if h == 0:
    #     h = 1
    a = w / h if h != 0 else 0
    if len(bbox) == 4:
        return [x, y, a, h]
    elif len(bbox) == 8:
        dx, dy, dw, dh = bbox[4], bbox[5], bbox[6], bbox[7]
        da =  a - (w - dw) / (h - dh) if h - dh != 0 else a
        return [x, y, a, h, dx, dy, da, dh]
    else:
        raise ValueError("Unknown bbox format.")

def xywh_to_xcycah(bbox):
    """
    Convert a bbox in [x, y, w, h] or [x, y, w, h, dx, dy, dw, dh] format to [xc, yc, a, h] or [xc, yc, a, h, dxc, dyc, da, dh] format.
    """
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    xc, yc = x + w / 2, y + h / 2
    # if h == 0:
    #     h = 1
    a = w / h if h != 0 else 0
    if len(bbox) == 4:
        return [xc, yc, a, h]
    elif len(bbox) == 8:
        dx, dy, dw, dh = bbox[4], bbox[5], bbox[6], bbox[7]
        dxc = dx + dw / 2
        dyc = dy + dh / 2
        da =  a - (w - dw) / (h - dh) if h - dh != 0 else a
        return [xc, yc, a, h, dxc, dyc, da, dh]
    else:
        raise ValueError("Unknown bbox format.")

def x1y1ah_to_xywh(bbox):
    """
    Convert a bbox in [x1, y1, a, h] or [x1, y1, a, h, dx1, dy1, da, dh] format to [x, y, w, h] or [x, y, w, h, dx, dy, dw, dh] format.
    """
    x, y, a, h = bbox[0], bbox[1], bbox[2], bbox[3]
    w = a * h
    if len(bbox) == 4:
        return [x, y, w, h]
    elif len(bbox) == 8:
        dx, dy, da, dh = bbox[4], bbox[5], bbox[6], bbox[7]
        dw = w - (a - da) * (h - dh)
        return [x, y, w, h, dx, dy, dw, dh]
    else:
        raise ValueError("Unknown bbox format.")

def xcycah_to_xywh(bbox):
    """
    Convert a bbox in [xc, yc, a, h] or [xc, yc, a, h, dxc, dyc, da, dh] format to [x, y, w, h] or [x, y, w, h, dx, dy, dw, dh] format.
    """
    xc, yc, a, h = bbox[0], bbox[1], bbox[2], bbox[3]
    w = a * h
    x, y = xc - w / 2, yc - h / 2
    if len(bbox) == 4:
        return [x, y, w, h]
    elif len(bbox) == 8:
        dxc, dyc, da, dh = bbox[4], bbox[5], bbox[6], bbox[7]
        dw = w - (a - da) * (h - dh)
        dx = dxc - dw / 2
        dy = dyc - dh / 2
        return [x, y, w, h, dx, dy, dw, dh]
    else:
        raise ValueError("Unknown bbox format.")

def normalize_xywh_bbox_by_img_size(box, original_wh, target_wh=DEFAULT_IMG_WH):
    """
    Normalize a bbox in [x, y, w, h] format to a 1024x1024 image.
    """
    if box == []:
        return []
    x, y, w, h = box[0], box[1], box[2], box[3]
    x_normalized = int(x / original_wh[0] * target_wh[0])
    y_normalized = int(y / original_wh[1] * target_wh[1])
    w_normalized = int(w / original_wh[0] * target_wh[0])
    h_normalized = int(h / original_wh[1] * target_wh[1])
    return [x_normalized, y_normalized, w_normalized, h_normalized]

def normalize_xywh_bbox_size(bbox_list, target_init_wh=DEFAULT_INIT_WH):
    """
    Normalize a list of bboxes in [x, y, w, h, dx, dy, dw, dh] format to the target initial width/height target_init_wh while keeping center points unchanged.
    """
    if bbox_list == []:
        return []
    init_w, init_h = bbox_list[0][2], bbox_list[0][3]
    w_factor = target_init_wh[0] / init_w if init_w != 0 else 0
    h_factor = target_init_wh[1] / init_h if init_h != 0 else 0
    normalized_bbox_list = []
    for i in range(len(bbox_list)):
        w_i = w_factor * bbox_list[i][2]
        h_i = h_factor * bbox_list[i][3]
        xc_i = bbox_list[i][0] + bbox_list[i][2] / 2
        yc_i = bbox_list[i][1] + bbox_list[i][3] / 2
        if len(bbox_list[i]) == 4:
            normalized_bbox_list.append([xc_i - w_i / 2, yc_i - h_i / 2, w_i, h_i])
        elif len(bbox_list[i]) == 8:
            # Normalize displacement and size changes
            dx_i = w_factor * bbox_list[i][4]
            dy_i = h_factor * bbox_list[i][5]
            dw_i = w_factor * bbox_list[i][6]
            dh_i = h_factor * bbox_list[i][7]
            normalized_bbox_list.append([xc_i - w_i / 2, yc_i - h_i / 2, w_i, h_i, dx_i, dy_i, dw_i, dh_i])
        else:
            raise ValueError("Unknown bbox format.")
    return normalized_bbox_list

def normalize_xywh_bbox_pos(bbox_list, target_init_pos=DEFAULT_INIT_POS):
    """
    Normalize a list of bboxes in [x, y, w, h, dx, dy, dw, dh] format to the target initial position target_init_pos while keeping width and height unchanged.
    """
    if bbox_list == []:
        return []
    init_xc, init_yc = bbox_list[0][0] + bbox_list[0][2] / 2, bbox_list[0][1] + bbox_list[0][3] / 2
    normalized_bbox_list = []
    for i in range(len(bbox_list)):
        x_i = bbox_list[i][0] - (init_xc - target_init_pos[0])
        y_i = bbox_list[i][1] - (init_yc - target_init_pos[1])
        if len(bbox_list[i]) == 4:
            normalized_bbox_list.append([x_i, y_i, bbox_list[i][2], bbox_list[i][3]])
        elif len(bbox_list[i]) == 8:
            # Keep displacement and size changes unchanged
            dx_i = bbox_list[i][4]
            dy_i = bbox_list[i][5]
            dw_i = bbox_list[i][6]
            dh_i = bbox_list[i][7]
            normalized_bbox_list.append([x_i, y_i, bbox_list[i][2], bbox_list[i][3], dx_i, dy_i, dw_i, dh_i])
        else:
            raise ValueError("Unknown bbox format.")
    return normalized_bbox_list

def recover_xywh_bbox_size(original_init_bbox, normalized_bbox, target_init_wh=DEFAULT_INIT_WH):
    """
    Recover a normalized bbox in [x, y, w, h, dx, dy, dw, dh] format to its original size.
    """
    if normalized_bbox == []:
        return []
    init_w, init_h = original_init_bbox[2], original_init_bbox[3]
    w_factor = target_init_wh[0] / init_w if init_w != 0 else 0
    h_factor = target_init_wh[1] / init_h if init_h != 0 else 0
    recovered_w = normalized_bbox[2] / w_factor if w_factor != 0 else 0
    recovered_h = normalized_bbox[3] / h_factor if h_factor != 0 else 0
    xc = normalized_bbox[0] + normalized_bbox[2] / 2
    yc = normalized_bbox[1] + normalized_bbox[3] / 2
    recovered_x = xc - recovered_w / 2
    recovered_y = yc - recovered_h / 2
    if len(normalized_bbox) == 4:
        return [recovered_x, recovered_y, recovered_w, recovered_h]
    elif len(normalized_bbox) == 8:
        recovered_dx = normalized_bbox[4] / w_factor if w_factor != 0 else 0
        recovered_dy = normalized_bbox[5] / h_factor if h_factor != 0 else 0
        recovered_dw = normalized_bbox[6] / w_factor if w_factor != 0 else 0
        recovered_dh = normalized_bbox[7] / h_factor if h_factor != 0 else 0
        return [recovered_x, recovered_y, recovered_w, recovered_h, recovered_dx, recovered_dy, recovered_dw, recovered_dh]
    else:
        raise ValueError("Unknown bbox format.")

def recover_xywh_bbox_pos(original_init_bbox, normalized_bbox, target_init_pos=DEFAULT_INIT_POS):
    """
    Recover a normalized bbox in [x, y, w, h, dx, dy, dw, dh] format to its original position.
    """
    if normalized_bbox == []:
        return []
    init_xc, init_yc = original_init_bbox[0] + original_init_bbox[2] / 2, original_init_bbox[1] + original_init_bbox[3] / 2
    x_i = normalized_bbox[0] + (init_xc - target_init_pos[0])
    y_i = normalized_bbox[1] + (init_yc - target_init_pos[1])
    if len(normalized_bbox) == 4:
        return [x_i, y_i, normalized_bbox[2], normalized_bbox[3]]
    elif len(normalized_bbox) == 8:
        return [x_i, y_i, normalized_bbox[2], normalized_bbox[3], normalized_bbox[4], normalized_bbox[5], normalized_bbox[6], normalized_bbox[7]]
    else:
        raise ValueError("Unknown bbox format.")

class DroneTrajectoryDataset300(Dataset):
    def __init__(
        self, 
        root_dir, 
        history_length=4, 
        only_complete_traj=True, 
        state_type="xywh", 
        augment=False, 
        add_acceleration=False,
        normalize_img_size=False,
        normalize_box_size=False,
        normalize_box_pos=False,
        targeted_json=None,
    ):
        """
        Initialize the dataset.
        root_dir: str, dataset root directory.
        history_length: int, historical frame length.
        """
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.json_num_per_video = 2
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if (self.normalize_box_size or self.normalize_box_pos) and self.add_acceleration:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in AntiUAV datasets yet when add_acceleration is True")
        self.targeted_json = targeted_json  # Target json file name; if None, use all json files
        self.json_to_image_wh = {
            "infrared.json": (640, 512),
            "visible.json": (1920, 1080),
            "IR_label.json": (640, 512),
            "RGB_label.json": (1920, 1080),
        }
        self.data = self._load_data(self.root_dir)
    
    def _apply_augmentation(self, historical_states, next_state):
        raise NotImplementedError("还没有针对速度和加速度做变换,引入normalize改动之后还没有适配")
    
    def _load_data(self, root_dir):
        """
        Load and filter all valid data to ensure the target always exists in each data group.
        Return: a list where each element is a tuple of (historical frames, target frame).
        """
        data = []
        subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for subfolder in subfolders:
            # Find two JSON files
            json_files = [f for f in os.listdir(subfolder) if f.endswith('.json')]
            if self.json_num_per_video != None and len(json_files) != self.json_num_per_video:
                continue  # Skip if the subfolder does not meet the format requirements

            for json_file in json_files:
                if self.targeted_json is not None and json_file != self.targeted_json:
                    continue  # If a target json file name is specified, load only that file
                json_path = os.path.join(subfolder, json_file)
                with open(json_path, 'r') as f:
                    content = json.load(f)

                exist = content["exist"]
                gt_rect = content["gt_rect"]
                image_wh = self.json_to_image_wh[json_file]
                if self.normalize_img_size:
                    for i in range(len(gt_rect)):
                        normalized_bbox = normalize_xywh_bbox_by_img_size(gt_rect[i], image_wh)
                        gt_rect[i] = normalized_bbox

                # Compute [dx, dy, dw, dh] and generate the full state [x, y, w, h, dx, dy, dw, dh]
                for i in range(1, len(gt_rect)):
                    # if exist[i] == 1 and exist[i - 1] == 1:  # Ensure the target exists in both the current and previous frames
                    prev_prev_state = gt_rect[i - 2] if (i - 2 >= 0 and gt_rect[i - 2] != []) else [0, 0, 0, 0]  # Use all zeros if the target does not exist two frames ago
                    prev_state = gt_rect[i - 1] if gt_rect[i - 1] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the previous frame
                    curr_state = gt_rect[i] if gt_rect[i] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the current frame
                    
                    velocity = [
                        curr_state[0] - prev_state[0],  # dx
                        curr_state[1] - prev_state[1],  # dy
                        curr_state[2] - prev_state[2],  # dw or da
                        curr_state[3] - prev_state[3],  # dh
                    ]
                    
                    # This velocity is currently the result over one second
                    
                    if self.add_acceleration:
                        prev_velocity = [
                            prev_state[0] - prev_prev_state[0],  # dx
                            prev_state[1] - prev_prev_state[1],  # dy
                            prev_state[2] - prev_prev_state[2],  # dw
                            prev_state[3] - prev_prev_state[3],  # dh
                            ]
                        acceleration = [
                            velocity[0] - prev_velocity[0],  # ddx
                            velocity[1] - prev_velocity[1],  # ddy
                            velocity[2] - prev_velocity[2],  # ddw
                            velocity[3] - prev_velocity[3],  # ddh
                        ]
                        full_state = curr_state + velocity + acceleration
                    else:
                        full_state = curr_state + velocity
                    
                    gt_rect[i] = full_state  # Update the current frame state

                # Generate historical frames and target frames
                for i in range(1, len(gt_rect) - self.history_length):
                    if all(exist[i-1:i + self.history_length + 1]) or not self.only_complete_traj:  # If complete trajectories are required, ensure both historical and target frames exist
                        states = gt_rect[i:i + self.history_length + 1]

                        # Normalize first
                        if self.normalize_box_size:
                            states = normalize_xywh_bbox_size(states)
                        if self.normalize_box_pos:
                            states = normalize_xywh_bbox_pos(states)
                        
                        # Then convert the state
                        for i in range(len(states)):
                            if states[i] == []:
                                states[i] = [0, 0, 0, 0, 0, 0, 0, 0]
                            if self.state_type == "x1y1ah":
                                states[i] = xywh_to_x1y1ah(states[i])
                            elif self.state_type == "xcycah":
                                states[i] = xywh_to_xcycah(states[i])
                            elif self.state_type == "xcycwh":
                                states[i] = xywh_to_xcycwh(states[i])
                            elif self.state_type != "xywh":
                                raise ValueError("Unknown state type.")
                                
                        historical_states = states[0:self.history_length]
                        next_state = states[self.history_length]
                        data.append((historical_states, next_state))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return data for the specified index.
        Return values: historical frames (Tensor) and target frame (Tensor).
        """
        historical_states, next_state = self.data[idx]
        # Apply transformations if data augmentation is enabled
        if self.augment:
            historical_states, next_state = self._apply_augmentation(historical_states, next_state)
        historical_states = torch.tensor(historical_states, dtype=torch.float32).flatten()  # Flatten historical frames
        next_state = torch.tensor(next_state, dtype=torch.float32)
        return historical_states, next_state

class DroneTrajectoryDataset410(DroneTrajectoryDataset300):
    def __init__(
        self, 
        root_dir, 
        history_length=4, 
        only_complete_traj=False, 
        state_type="xywh", 
        augment=False, 
        add_acceleration=False,
        normalize_img_size=False,
        normalize_box_size=False,
        normalize_box_pos=False,
        targeted_json=None,
    ):
        super().__init__(root_dir, history_length=history_length, only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
        self.json_num_per_video = 1
        self.json_to_image_wh = {
            "IR_label.json": (640, 512),
        }
        self.data = self._load_data(self.root_dir)
    
class DroneTrajectoryDataset600(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        super().__init__(root_dir, history_length=history_length, only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
        self.json_num_per_video = 1
        self.json_to_image_wh = {
            "IR_label.json": (640, 512),
        }
        self.data = self._load_data(self.root_dir)  
          
class DroneTrajectoryDatasetWorkshop(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        super().__init__(root_dir, history_length=history_length, only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
        self.json_num_per_video = 1
        self.json_to_image_wh = {
            "IR_label.json": (640, 512),
        }
        self.data = self._load_data(self.root_dir)

class CombinedDroneTrajectoryDatasetAntiUAV(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if self.normalize_box_size or self.normalize_box_pos:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in AntiUAV datasets yet")
        self.targeted_json = targeted_json  # Target json file name; if None, use all json files
        self.json_num_per_video = None
        self.json_to_image_wh = {
            "infrared.json": (640, 512),
            "visible.json": (1920, 1080),
            "IR_label.json": (640, 512),
            "RGB_label.json": (1920, 1080),
        }
        self.data = []
        for one_root_dir in self.root_dir:
            self.data += self._load_data(one_root_dir)
            
class DroneTrajectoryDatasetDronevsBird(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        super().__init__(root_dir, history_length=history_length, only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
        self.data = self._load_data(self.root_dir)
    
    def _load_data(self):
        raise NotImplementedError

class DroneTrajectoryDatasetDUT(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None, GT_dir="Anti-UAV-Tracking-V0GT"):
        super().__init__(root_dir, history_length=history_length, only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
        self.GT_dir = GT_dir
        self.data = self._load_data(self.root_dir)
    
    def _load_data(self, root_dir):
        """
        Load and filter all valid data to ensure the target always exists in each data group.
        Return: a list where each element is a tuple of (historical frames, target frame).
        """
        data = []
        txts = [os.path.join(root_dir, self.GT_dir, d) for d in os.listdir(os.path.join(root_dir, self.GT_dir)) if d.endswith('.txt')]
        raise NotImplementedError
        # need to ignore [-100 -100 -100 -100]

class TrajectoryDatasetTrackingNet(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.json_num_per_video = 2
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size to 256*256
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if (self.normalize_box_size or self.normalize_box_pos) and self.add_acceleration:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in TrackingNet datasets yet when add_acceleration is True")
        self.targeted_json = targeted_json  # Target json file name; if None, use all json files
        self.data = self._load_data(self.root_dir)
        if augment:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data augmentation yet.")
        if normalize_img_size:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data normalization by image size yet.")

    def _load_data(self, root_dir):
        """
        Load and filter all valid data to ensure the target always exists in each data group.
        Return: a list where each element is a tuple of (historical frames, target frame).
        """
        data = []
        subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # print(f'Loading files from {self.root_dir} ...')
        
        for subfolder in subfolders:
            print(f'Loading files from {subfolder} ...')
            anno_dir = os.path.join(subfolder, "anno")
            if not os.path.exists(anno_dir):
                continue
            # Find txt files
            txt_files = [f for f in os.listdir(anno_dir) if f.endswith('.txt')]

            for txt_file in tqdm(txt_files):
                txt_path = os.path.join(anno_dir, txt_file)
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                gt_rect = []
                exist = [0] * len(lines)
                for line in lines:
                    x, y, w, h = map(lambda x: float(x), line.split(','))
                    gt_rect.append([x, y, w, h])
                
                # Compute [dx, dy, dw, dh] and generate the full state [x, y, w, h, dx, dy, dw, dh]
                for i in range(1, len(gt_rect)):
                    # if exist[i] == 1 and exist[i - 1] == 1:  # Ensure the target exists in both the current and previous frames
                    prev_prev_state = gt_rect[i - 2] if (i - 2 >= 0 and gt_rect[i - 2] != []) else [0, 0, 0, 0]  # Use all zeros if the target does not exist two frames ago
                    prev_state = gt_rect[i - 1] if gt_rect[i - 1] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the previous frame
                    curr_state = gt_rect[i] if gt_rect[i] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the current frame
                    if gt_rect[i] != [] and gt_rect[i] != [-100, -100, -100, -100] and gt_rect[i] != [0, 0, 0, 0]:
                        exist[i] = 1
                    else:
                        exist[i] = 0
                    
                    velocity = [
                        curr_state[0] - prev_state[0],  # dx
                        curr_state[1] - prev_state[1],  # dy
                        curr_state[2] - prev_state[2],  # dw
                        curr_state[3] - prev_state[3],  # dh
                    ]
                    
                    # This velocity is currently the result over one second
                    
                    if self.add_acceleration:
                        prev_velocity = [
                            prev_state[0] - prev_prev_state[0],  # dx
                            prev_state[1] - prev_prev_state[1],  # dy
                            prev_state[2] - prev_prev_state[2],  # dw
                            prev_state[3] - prev_prev_state[3],  # dh
                            ]
                        acceleration = [
                            velocity[0] - prev_velocity[0],  # ddx
                            velocity[1] - prev_velocity[1],  # ddy
                            velocity[2] - prev_velocity[2],  # ddw
                            velocity[3] - prev_velocity[3],  # ddh
                        ]
                        full_state = curr_state + velocity + acceleration
                    else:
                        full_state = curr_state + velocity
                    
                    gt_rect[i] = full_state  # Update the current frame state

                # Generate historical frames and target frames
                for i in range(1, len(gt_rect) - self.history_length):
                    if all(exist[i-1:i + self.history_length + 1]) or not self.only_complete_traj:  # If complete trajectories are required, ensure both historical and target frames exist
                        states = gt_rect[i:i + self.history_length + 1]

                        # Normalize first
                        if self.normalize_box_size:
                            states = normalize_xywh_bbox_size(states)
                        if self.normalize_box_pos:
                            states = normalize_xywh_bbox_pos(states)
                        
                        # Then convert the state
                        for i in range(len(states)):
                            if states[i] == []:
                                states[i] = [0, 0, 0, 0, 0, 0, 0, 0]
                            if self.state_type == "x1y1ah":
                                states[i] = xywh_to_x1y1ah(states[i])
                            elif self.state_type == "xcycah":
                                states[i] = xywh_to_xcycah(states[i])
                            elif self.state_type == "xcycwh":
                                states[i] = xywh_to_xcycwh(states[i])
                            elif self.state_type != "xywh":
                                raise ValueError("Unknown state type.")
                                
                        historical_states = states[0:self.history_length]
                        next_state = states[self.history_length]
                        data.append((historical_states, next_state))

        return data

class TrajectoryDatasetLaSOT(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.json_num_per_video = 2
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size to 256*256
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if (self.normalize_box_size or self.normalize_box_pos) and self.add_acceleration:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in TrackingNet datasets yet when add_acceleration is True")
        self.targeted_json = targeted_json  # Target json file name; if None, use all json files
        self.data = self._load_data(self.root_dir)
        if augment:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data augmentation yet.")
        if normalize_img_size:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data normalization by image size yet.")

    def _load_data(self, root_dir):
        """
        Load and filter all valid data to ensure the target always exists in each data group.
        Return: a list where each element is a tuple of (historical frames, target frame).
        """
        data = []

        # Find txt files
        txt_files = [f for f in os.listdir(root_dir) if f.endswith('.txt')]

        for txt_file in tqdm(txt_files):
            txt_path = os.path.join(root_dir, txt_file)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            absent_txt_path = os.path.join(root_dir, 'absent', txt_file)
            with open(absent_txt_path, 'r') as f:
                absent_lines = f.readlines()
            assert len(lines) == len(absent_lines), f'{txt_file} has {len(lines)} lines, but {absent_txt_path} has {len(absent_lines)} lines'
            gt_rect = []
            exist = []
            for i,line in enumerate(lines):
                if absent_lines[i].split('\n')[0] == '1':
                    exist.append(0)
                    x, y, w, h = [0, 0, 0, 0]
                elif absent_lines[i].split('\n')[0] == '0':
                    exist.append(1)
                    x, y, w, h = map(lambda x: float(x), line.split(','))
                else:
                    raise ValueError(f'{absent_txt_path} has invalid line {i}: {absent_lines[i]}')
                gt_rect.append([x, y, w, h])
            
            # Compute [dx, dy, dw, dh] and generate the full state [x, y, w, h, dx, dy, dw, dh]
            for i in range(1, len(gt_rect)):
                # if exist[i] == 1 and exist[i - 1] == 1:  # Ensure the target exists in both the current and previous frames
                prev_prev_state = gt_rect[i - 2] if (i - 2 >= 0 and gt_rect[i - 2] != []) else [0, 0, 0, 0]  # Use all zeros if the target does not exist two frames ago
                prev_state = gt_rect[i - 1] if gt_rect[i - 1] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the previous frame
                curr_state = gt_rect[i] if gt_rect[i] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the current frame
                
                velocity = [
                    curr_state[0] - prev_state[0],  # dx
                    curr_state[1] - prev_state[1],  # dy
                    curr_state[2] - prev_state[2],  # dw
                    curr_state[3] - prev_state[3],  # dh
                ]
                
                # This velocity is currently the result over one second
                
                if self.add_acceleration:
                    prev_velocity = [
                        prev_state[0] - prev_prev_state[0],  # dx
                        prev_state[1] - prev_prev_state[1],  # dy
                        prev_state[2] - prev_prev_state[2],  # dw
                        prev_state[3] - prev_prev_state[3],  # dh
                        ]
                    acceleration = [
                        velocity[0] - prev_velocity[0],  # ddx
                        velocity[1] - prev_velocity[1],  # ddy
                        velocity[2] - prev_velocity[2],  # ddw
                        velocity[3] - prev_velocity[3],  # ddh
                    ]
                    full_state = curr_state + velocity + acceleration
                else:
                    full_state = curr_state + velocity
                
                gt_rect[i] = full_state  # Update the current frame state

            # Generate historical frames and target frames
            for i in range(1, len(gt_rect) - self.history_length):
                if all(exist[i-1:i + self.history_length + 1]) or not self.only_complete_traj:  # If complete trajectories are required, ensure both historical and target frames exist
                    states = gt_rect[i:i + self.history_length + 1]

                    # Normalize first
                    if self.normalize_box_size:
                        states = normalize_xywh_bbox_size(states)
                    if self.normalize_box_pos:
                        states = normalize_xywh_bbox_pos(states)
                    
                    # Then convert the state
                    for i in range(len(states)):
                        if states[i] == []:
                            states[i] = [0, 0, 0, 0, 0, 0, 0, 0]
                        if self.state_type == "x1y1ah":
                            states[i] = xywh_to_x1y1ah(states[i])
                        elif self.state_type == "xcycah":
                            states[i] = xywh_to_xcycah(states[i])
                        elif self.state_type == "xcycwh":
                            states[i] = xywh_to_xcycwh(states[i])
                        elif self.state_type != "xywh":
                            raise ValueError("Unknown state type.")
                            
                    historical_states = states[0:self.history_length]
                    next_state = states[self.history_length]
                    data.append((historical_states, next_state))

        return data

class TrajectoryDatasetYouTubeVOS2019(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.json_num_per_video = 2
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size to 256*256
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if (self.normalize_box_size or self.normalize_box_pos) and self.add_acceleration:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in TrackingNet datasets yet when add_acceleration is True")
        self.targeted_json = targeted_json  # Target json file name; if None, use all json files
        self.data = self._load_data(self.root_dir)
        if augment:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data augmentation yet.")
        if normalize_img_size:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data normalization by image size yet.")

    def _load_data(self, root_dir):
        """
        Load and filter all valid data to ensure the target always exists in each data group.
        Return: a list where each element is a tuple of (historical frames, target frame).
        """
        data = []

        # Find json files
        json_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]

        for json_file in tqdm(json_files):
            json_path = os.path.join(root_dir, json_file)
            with open(json_path, 'r') as f:
                content = json.load(f)
            exist = content["exist"]
            gt_rect = content["gt_rect"]
            
            # Compute [dx, dy, dw, dh] and generate the full state [x, y, w, h, dx, dy, dw, dh]
            for i in range(1, len(gt_rect)):
                # if exist[i] == 1 and exist[i - 1] == 1:  # Ensure the target exists in both the current and previous frames
                prev_prev_state = gt_rect[i - 2] if (i - 2 >= 0 and gt_rect[i - 2] != []) else [0, 0, 0, 0]  # Use all zeros if the target does not exist two frames ago
                prev_state = gt_rect[i - 1] if gt_rect[i - 1] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the previous frame
                curr_state = gt_rect[i] if gt_rect[i] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the current frame
                
                velocity = [
                    curr_state[0] - prev_state[0],  # dx
                    curr_state[1] - prev_state[1],  # dy
                    curr_state[2] - prev_state[2],  # dw
                    curr_state[3] - prev_state[3],  # dh
                ]
                
                # This velocity is currently the result over one second
                
                if self.add_acceleration:
                    prev_velocity = [
                        prev_state[0] - prev_prev_state[0],  # dx
                        prev_state[1] - prev_prev_state[1],  # dy
                        prev_state[2] - prev_prev_state[2],  # dw
                        prev_state[3] - prev_prev_state[3],  # dh
                        ]
                    acceleration = [
                        velocity[0] - prev_velocity[0],  # ddx
                        velocity[1] - prev_velocity[1],  # ddy
                        velocity[2] - prev_velocity[2],  # ddw
                        velocity[3] - prev_velocity[3],  # ddh
                    ]
                    full_state = curr_state + velocity + acceleration
                else:
                    full_state = curr_state + velocity
                
                gt_rect[i] = full_state  # Update the current frame state

            # Generate historical frames and target frames
            for i in range(1, len(gt_rect) - self.history_length):
                if all(exist[i-1:i + self.history_length + 1]) or not self.only_complete_traj:  # If complete trajectories are required, ensure both historical and target frames exist
                    states = gt_rect[i:i + self.history_length + 1]

                    # Normalize first
                    if self.normalize_box_size:
                        states = normalize_xywh_bbox_size(states)
                    if self.normalize_box_pos:
                        states = normalize_xywh_bbox_pos(states)
                    
                    # Then convert the state
                    for i in range(len(states)):
                        if states[i] == []:
                            states[i] = [0, 0, 0, 0, 0, 0, 0, 0]
                        if self.state_type == "x1y1ah":
                            states[i] = xywh_to_x1y1ah(states[i])
                        elif self.state_type == "xcycah":
                            states[i] = xywh_to_xcycah(states[i])
                        elif self.state_type == "xcycwh":
                            states[i] = xywh_to_xcycwh(states[i])
                        elif self.state_type != "xywh":
                            raise ValueError("Unknown state type.")
                            
                    historical_states = states[0:self.history_length]
                    next_state = states[self.history_length]
                    data.append((historical_states, next_state))

        return data

class TrajectoryDatasetMOSEv2(TrajectoryDatasetYouTubeVOS2019):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.json_num_per_video = 2
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size to 256*256
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if (self.normalize_box_size or self.normalize_box_pos) and self.add_acceleration:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in TrackingNet datasets yet when add_acceleration is True")
        self.targeted_json = targeted_json  # Target json file name; if None, use all json files
        self.data = self._load_data(self.root_dir)
        if augment:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data augmentation yet.")
        if normalize_img_size:
            raise NotImplementedError("TrajectoryDatasetTrackingNet has not support data normalization by image size yet.")

class TrajectoryDatasetUniversalUAVTracking(DroneTrajectoryDataset300):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if self.normalize_box_size or self.normalize_box_pos:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in Universal UAV Tracking datasets yet")
        self.data = self._load_data(self.root_dir)
        if augment:
            raise NotImplementedError("TrajectoryDatasetUniversalUAVTracking has not support data augmentation yet.")
        if normalize_img_size:
            raise NotImplementedError("TrajectoryDatasetUniversalUAVTracking has not support data normalization by image size yet. Using raw data instead.")
    
    def _load_data(self, root_dir):
        """
        Load and filter all valid data to ensure the target always exists in each data group.
        Return: a list where each element is a tuple of (historical frames, target frame).
        """
        data = []
        # Find txt files
        anno_dir = root_dir
        txt_files = [f for f in os.listdir(anno_dir) if f.endswith('.txt')]

        for txt_file in tqdm(txt_files):
            txt_path = os.path.join(anno_dir, txt_file)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            gt_rect = []
            exist = []
            for line in lines:
                if 'nan' in line or 'NaN' in line:
                    gt_rect.append([0, 0, 0, 0])
                    exist.append(0)
                else:
                    x, y, w, h = map(lambda x: int(float(x)), line.split(','))
                    gt_rect.append([x, y, w, h])
                    exist.append(1)
            
            # Compute [dx, dy, dw, dh] and generate the full state [x, y, w, h, dx, dy, dw, dh]
            for i in range(1, len(gt_rect)):
                # if exist[i] == 1 and exist[i - 1] == 1:  # Ensure the target exists in both the current and previous frames
                prev_prev_state = gt_rect[i - 2] if (i - 2 >= 0 and gt_rect[i - 2] != []) else [0, 0, 0, 0]  # Use all zeros if the target does not exist two frames ago
                prev_state = gt_rect[i - 1] if gt_rect[i - 1] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the previous frame
                curr_state = gt_rect[i] if gt_rect[i] != [] else [0, 0, 0, 0]  # Use all zeros if the target does not exist in the current frame
                
                velocity = [
                    curr_state[0] - prev_state[0],  # dx
                    curr_state[1] - prev_state[1],  # dy
                    curr_state[2] - prev_state[2],  # dw
                    curr_state[3] - prev_state[3],  # dh
                ]
                
                # This velocity is currently the result over one second
                
                if self.add_acceleration:
                    prev_velocity = [
                        prev_state[0] - prev_prev_state[0],  # dx
                        prev_state[1] - prev_prev_state[1],  # dy
                        prev_state[2] - prev_prev_state[2],  # dw
                        prev_state[3] - prev_prev_state[3],  # dh
                        ]
                    acceleration = [
                        velocity[0] - prev_velocity[0],  # ddx
                        velocity[1] - prev_velocity[1],  # ddy
                        velocity[2] - prev_velocity[2],  # ddw
                        velocity[3] - prev_velocity[3],  # ddh
                    ]
                    full_state = curr_state + velocity + acceleration
                else:
                    full_state = curr_state + velocity
                
                gt_rect[i] = full_state  # Update the current frame state

                # Generate historical frames and target frames
                for i in range(1, len(gt_rect) - self.history_length):
                    if all(exist[i-1:i + self.history_length + 1]) or not self.only_complete_traj:  # If complete trajectories are required, ensure both historical and target frames exist
                        states = gt_rect[i:i + self.history_length + 1]

                        # Normalize first
                        if self.normalize_box_size:
                            states = normalize_xywh_bbox_size(states)
                        if self.normalize_box_pos:
                            states = normalize_xywh_bbox_pos(states)
                        
                        # Then convert the state
                        for i in range(len(states)):
                            if states[i] == []:
                                states[i] = [0, 0, 0, 0, 0, 0, 0, 0]
                            if self.state_type == "x1y1ah":
                                states[i] = xywh_to_x1y1ah(states[i])
                            elif self.state_type == "xcycah":
                                states[i] = xywh_to_xcycah(states[i])
                            elif self.state_type == "xcycwh":
                                states[i] = xywh_to_xcycwh(states[i])
                            elif self.state_type != "xywh":
                                raise ValueError("Unknown state type.")
                                
                        historical_states = states[0:self.history_length]
                        next_state = states[self.history_length]
                        data.append((historical_states, next_state))

        return data


class CombinedDroneTrajectoryDatasetUAVTracking(TrajectoryDatasetUniversalUAVTracking):
    def __init__(self, root_dir, history_length=4, only_complete_traj=False, state_type="xywh", augment=False, add_acceleration=False, normalize_img_size=False, normalize_box_size=False, normalize_box_pos=False, targeted_json=None):
        self.root_dir = root_dir
        self.history_length = history_length
        self.only_complete_traj = only_complete_traj
        self.state_type = state_type
        self.augment = augment
        self.add_acceleration = add_acceleration  # Whether to add acceleration information
        self.normalize_img_size = normalize_img_size  # Whether to normalize to a 1024*1024 image
        self.normalize_box_size = normalize_box_size  # Whether to normalize the initial box size
        self.normalize_box_pos = normalize_box_pos  # Whether to normalize the initial box position
        if (self.normalize_box_size or self.normalize_box_pos) and self.add_acceleration:
            raise NotImplementedError("normalize_box_size and normalize_box_pos are not implemented in Universal UAV Tracking datasets yet when add_acceleration is True.")
        self.data = []
        raise NotImplementedError("CombinedDroneTrajectoryDatasetUAVTracking has not support data loading yet.")
        # for one_root_dir in self.root_dir:
        #     self.data += self._load_data(one_root_dir)
            
def get_dataloader(
    root_dir, 
    history_length=4, 
    only_complete_traj=True, 
    state_type="xywh", 
    batch_size=32, 
    shuffle=True, 
    augment=False, 
    add_acceleration=False,
    normalize_img_size=False,
    normalize_box_size=False,
    normalize_box_pos=False,
    targeted_json=None
):
    """
    Create a DataLoader.
    root_dir: str, dataset root directory.
    history_length: int, historical frame length.
    batch_size: int, batch size.
    shuffle: bool, whether to shuffle the data.
    augment: bool, whether to perform data augmentation.
    """
    if isinstance(root_dir, list) and len(root_dir) == 1:
        root_dir = root_dir[0]
    # If root_dir is a list, create a CombinedDroneTrajectoryDataset
    if isinstance(root_dir, list):
        # dataset = CombinedDroneTrajectoryDatasetAntiUAV(root_dir, history_length=history_length, \
        dataset = CombinedDroneTrajectoryDatasetUAVTracking(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/Anti-UAV300/" in root_dir:
        dataset = DroneTrajectoryDataset300(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/Anti-UAV410/" in root_dir:
        dataset = DroneTrajectoryDataset410(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/Anti-UAV600/" in root_dir:
        dataset = DroneTrajectoryDataset600(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/workshop/" in root_dir:
        dataset = DroneTrajectoryDatasetWorkshop(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/Drone_vs_Bird/" in root_dir:
        dataset = DroneTrajectoryDatasetDronevsBird(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/DUT/" in root_dir:
        dataset = DroneTrajectoryDatasetDUT(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/TrackingNet/" in root_dir:
        dataset = TrajectoryDatasetTrackingNet(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/LaSOT/" in root_dir:
        dataset = TrajectoryDatasetLaSOT(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/YouTubeVOS2019/" in root_dir:
        dataset = TrajectoryDatasetYouTubeVOS2019(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/MOSEv2/" in root_dir:
        dataset = TrajectoryDatasetMOSEv2(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    elif "/VisDrone2019-SOT/" in root_dir \
        or "/UAV123/" in root_dir \
            or "/UAV123_10fps/" in root_dir \
                or "/UAVDT/" in root_dir \
                    or "/UAVTrack112/" in root_dir:
        dataset = TrajectoryDatasetUniversalUAVTracking(root_dir, history_length=history_length, \
            only_complete_traj=only_complete_traj, state_type=state_type, augment=augment, \
                add_acceleration=add_acceleration, normalize_img_size=normalize_img_size, \
                    normalize_box_size=normalize_box_size, normalize_box_pos=normalize_box_pos, targeted_json=targeted_json)
    else:
        raise ValueError(f"Not implemented dataset or unrecognized dataset path: {root_dir}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class GPUDatasetWrapper(Dataset):
    """
    Wrapper class for preloading a dataset into GPU memory.
    """
    def __init__(self, dataset, device='cuda', pin_memory=True):
        """
        Initialize the GPU dataset wrapper.
        
        Args:
            dataset: Original dataset
            device: Target device ('cuda' or 'cpu')
            pin_memory: Whether to use pinned memory
        """
        self.dataset = dataset
        self.device = device
        self.pin_memory = pin_memory
        
        print(f"正在将数据集预加载到 {device}...")
        self._preload_data()
        print(f"数据集预加载完成！共 {len(self.dataset)} 个样本")
    
    def _preload_data(self):
        """Preload all data to the GPU."""
        self.historical_states_gpu = []
        self.next_states_gpu = []
        
        # Use tqdm to show progress
        for i in tqdm(range(len(self.dataset)), desc="预加载数据到GPU"):
            historical_states, next_state = self.dataset[i]
            
            # Move data to the GPU
            if self.device == 'cuda':
                historical_states_gpu = historical_states.to(self.device, non_blocking=True)
                next_state_gpu = next_state.to(self.device, non_blocking=True)
            else:
                historical_states_gpu = historical_states
                next_state_gpu = next_state
            
            self.historical_states_gpu.append(historical_states_gpu)
            self.next_states_gpu.append(next_state_gpu)
        
        # Clean up CPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Return data directly from the GPU without data transfer."""
        return self.historical_states_gpu[idx], self.next_states_gpu[idx]
    
    def get_memory_usage(self):
        """Get GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return allocated, cached
        return 0, 0

def get_gpu_dataloader(
    root_dir, 
    history_length=4, 
    only_complete_traj=True, 
    state_type="xywh", 
    batch_size=32, 
    shuffle=True, 
    augment=False, 
    add_acceleration=False,
    normalize_img_size=False,
    normalize_box_size=False,
    normalize_box_pos=False,
    targeted_json=None,
    device='cuda',
    pin_memory=True
):
    """
    Create a DataLoader preloaded to the GPU.
    
    Args:
        root_dir: Dataset root directory
        history_length: Historical frame length
        only_complete_traj: Whether to use only complete trajectories
        state_type: State type
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        augment: Whether to perform data augmentation
        add_acceleration: Whether to add acceleration
        normalize_img_size: Whether to normalize image size
        normalize_box_size: Whether to normalize box size
        normalize_box_pos: Whether to normalize box position
        targeted_json: Target json file
        device: Target device
        pin_memory: Whether to use pinned memory
    
    Returns:
        DataLoader: DataLoader preloaded to the GPU
    """
    # First create the original dataset
    original_dataset = get_dataloader(
        root_dir, 
        history_length=history_length,
        only_complete_traj=only_complete_traj,
        state_type=state_type,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle first; handle it in the GPU wrapper
        augment=augment,
        add_acceleration=add_acceleration,
        normalize_img_size=normalize_img_size,
        normalize_box_size=normalize_box_size,
        normalize_box_pos=normalize_box_pos,
        targeted_json=targeted_json
    ).dataset
    
    # Create the GPU wrapper
    gpu_dataset = GPUDatasetWrapper(original_dataset, device=device, pin_memory=pin_memory)
    
    # Create the DataLoader
    dataloader = DataLoader(
        gpu_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=False,  # Data is already on the GPU, so pin_memory is unnecessary
        num_workers=0  # GPU data does not need multi-process loading
    )
    
    return dataloader
