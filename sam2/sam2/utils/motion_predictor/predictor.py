import torch
import torch.nn as nn
try:
    from traj_dataset import normalize_xywh_bbox_size, normalize_xywh_bbox_pos, \
        xywh_to_x1y1ah, xywh_to_xcycah, xywh_to_xcycwh, x1y1ah_to_xywh, xcycah_to_xywh, xcycwh_to_xywh, \
        recover_xywh_bbox_size, recover_xywh_bbox_pos
except ImportError:
    from sam2.utils.motion_predictor.traj_dataset import normalize_xywh_bbox_size, normalize_xywh_bbox_pos, \
        xywh_to_x1y1ah, xywh_to_xcycah, xywh_to_xcycwh, x1y1ah_to_xywh, xcycah_to_xywh, xcycwh_to_xywh, \
        recover_xywh_bbox_size, recover_xywh_bbox_pos

class MLPMarkovModel(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=64, history_length=4):
        super(MLPMarkovModel, self).__init__()
        input_dim = history_length * state_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state):
        """
        Take historical states and output the predicted next state.
        """
        return self.model(state)

class RNNMarkovModel(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=64, history_length=4, num_layers=1):
        super(RNNMarkovModel, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x is the flattened historical state (batch_size, history_length*state_dim)
        batch_size = x.size(0) if x.dim() == 2 else 1
        
        # Reshape to sequence input format (batch_size, seq_len, input_size)
        x = x.view(batch_size, self.history_length, self.state_dim)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass
        # out, _ = self.rnn(x, (h0, c0))
        out, _ = self.rnn(x, h0)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Use the fully connected layer to get the prediction
        out = self.fc(out)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        return out

class LSTMMarkovModel(RNNMarkovModel):
    def __init__(self, state_dim=8, hidden_dim=64, history_length=4, num_layers=1):
        super(LSTMMarkovModel, self).__init__(state_dim, hidden_dim, history_length, num_layers)
        self.rnn = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, inp_x):
        # x is the flattened historical state (batch_size, history_length*state_dim)
        batch_size = inp_x.size(0) if inp_x.dim() == 2 else 1
        
        # Reshape to sequence input format (batch_size, seq_len, input_size)
        x = inp_x.view(batch_size, self.history_length, self.state_dim)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass
        out, _ = self.rnn(x, (h0, c0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Use the fully connected layer to get the prediction
        out = self.fc(out)
        if inp_x.dim() != 2:
            out = out.squeeze(0)
        return out
    
class MotionPredictor:
    def __init__(self, model_type="MarkovModel", state_type="xywh", state_dim=8, hidden_dim=64, history_size=4, rnn_layers=1, \
        normalize_box_size=False, normalize_box_pos=False, reverse=False):
        """
        Initialize the model.
        state_dim: int, state dimension.
        hidden_dim: int, hidden layer dimension.
        history_length: int, historical trajectory length.
        """
        self.state_type = state_type
        self.history_size = history_size
        if model_type == "MLPMarkovModel" or model_type == "MarkovModel":
            self.model = MLPMarkovModel(state_dim=state_dim, hidden_dim=hidden_dim, history_length=history_size)
        elif model_type == "RNNMarkovModel":
            self.model = RNNMarkovModel(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                history_length=history_size,
                num_layers=rnn_layers
            )
        elif model_type == "LSTMMarkovModel":
            self.model = LSTMMarkovModel(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                history_length=history_size,
                num_layers=rnn_layers
            )
        else:
            raise ValueError("unknown model type")
        self.history_bank = []  # Initialize historical trajectory bank
        self.prediction_bank = []
        self.state_dim = state_dim
        assert self.state_dim in [8], "state_dim should be 8 (without acceleration). No longer support 12 (with acceleration)"
        self.normalize_box_size = normalize_box_size
        self.normalize_box_pos = normalize_box_pos
        self.reverse = reverse

    def update_history_bank(self, new_state):
        """
        Update the historical trajectory bank.
        new_state: Tensor, shape (state_dim,).
        """
        assert len(new_state) == self.state_dim, "new_state should have the same dimension as state_dim, provide a full state with velocity = 0 if not provided"
        if not self.reverse:
            self.history_bank.append(new_state)
            if len(self.history_bank) > self.history_size:
                self.history_bank.pop(0)
        else:
                # Update the velocity of the previous box
            if len(self.history_bank) > 0:
                self.history_bank[0][4:] = [self.history_bank[0][i] - new_state[i] for i in range(4)]
            self.history_bank.insert(0, new_state)
            if len(self.history_bank) > self.history_size:
                self.history_bank.pop()
    
    def _update_prediction_bank(self, new_pred):
        """
        Update the historical trajectory bank.
        new_state: Tensor, shape (state_dim,).
        """
        if self.reverse:
            raise NotImplementedError("reverse motion prediction is not implemented for prediction bank")
        self.prediction_bank.append(new_pred)
        if len(self.prediction_bank) > self.history_size:
            self.prediction_bank.pop(0)
    
    def clear_banks(self):
        """
        Clear the historical trajectory bank.
        """
        self.history_bank = []
        self.prediction_bank = []
    
    def history_bank_length(self):
        """
        Return the historical trajectory bank length.
        """
        return len(self.history_bank)
    
    def _normalize_bbox_size(self, box_list):
        """
        Normalize the size of the box sequence.
        """
        # First convert all boxes to xywh format
        if self.state_type == "xywh":
            xywh_box_list = box_list
        elif self.state_type == "xcycwh":
            xywh_box_list = [xcycwh_to_xywh(box) for box in box_list]
        elif self.state_type == "xcycah":
            xywh_box_list = [xcycah_to_xywh(box) for box in box_list]
        elif self.state_type == "x1y1ah":
            xywh_box_list = [x1y1ah_to_xywh(box) for box in box_list]
        else:
            raise ValueError("unknown state type")
        
        # Then normalize
        normalized_xywh_box_list = normalize_xywh_bbox_size(xywh_box_list)
        
        # Then convert back to the original format
        if self.state_type == "xywh":
            return normalized_xywh_box_list
        elif self.state_type == "xcycwh":
            return [xywh_to_xcycwh(box) for box in normalized_xywh_box_list]
        elif self.state_type == "xcycah":
            return [xywh_to_xcycah(box) for box in normalized_xywh_box_list]
        elif self.state_type == "x1y1ah":
            return [xywh_to_x1y1ah(box) for box in normalized_xywh_box_list]
        else:
            raise ValueError("unknown state type")

    
    def _normalize_bbox_pos(self, box_list):
        """
        Normalize the positions of the box sequence.
        """
        # First convert all boxes to xywh format
        if self.state_type == "xywh":
            xywh_box_list = box_list
        elif self.state_type == "xcycwh":
            xywh_box_list = [xcycwh_to_xywh(box) for box in box_list]
        elif self.state_type == "xcycah":
            xywh_box_list = [xcycah_to_xywh(box) for box in box_list]
        elif self.state_type == "x1y1ah":
            xywh_box_list = [x1y1ah_to_xywh(box) for box in box_list]
        else:
            raise ValueError("unknown state type")
        
        # Then normalize
        normalized_xywh_box_list = normalize_xywh_bbox_pos(xywh_box_list)
        
        # Then convert back to the original format
        if self.state_type == "xywh":
            return normalized_xywh_box_list
        elif self.state_type == "xcycwh":
            return [xywh_to_xcycwh(box) for box in normalized_xywh_box_list]
        elif self.state_type == "xcycah":
            return [xywh_to_xcycah(box) for box in normalized_xywh_box_list]
        elif self.state_type == "x1y1ah":
            return [xywh_to_x1y1ah(box) for box in normalized_xywh_box_list]
        else:
            raise ValueError("unknown state type")

    def _recover_bbox_size(self, init_state, output_state):
        """
        Recover the box size.
        """
        # First convert both boxes to xywh format
        if self.state_type == "xywh":
            xywh_output_state = output_state
            xywh_init_state = init_state
        elif self.state_type == "xcycwh":
            xywh_output_state = xcycwh_to_xywh(output_state)
            xywh_init_state = xcycwh_to_xywh(init_state)
        elif self.state_type == "xcycah":
            xywh_output_state = xcycah_to_xywh(output_state)
            xywh_init_state = xcycah_to_xywh(init_state)
        elif self.state_type == "x1y1ah":
            xywh_output_state = x1y1ah_to_xywh(output_state)
            xywh_init_state = x1y1ah_to_xywh(init_state)
        else:
            raise ValueError("unknown state type")
        # Then recover
        recovered_xywh_output_state = recover_xywh_bbox_size(xywh_init_state, xywh_output_state)
        # Then convert back to the original format
        if self.state_type == "xywh":
            return recovered_xywh_output_state
        elif self.state_type == "xcycwh":
            return xywh_to_xcycwh(recovered_xywh_output_state)
        elif self.state_type == "xcycah":
            return xywh_to_xcycah(recovered_xywh_output_state)
        elif self.state_type == "x1y1ah":
            return xywh_to_x1y1ah(recovered_xywh_output_state)
        else:
            raise ValueError("unknown state type")
    
    def _recover_bbox_pos(self, init_state, output_state):
        """
        Recover the box position.
        """
        # First convert both boxes to xywh format
        if self.state_type == "xywh":
            xywh_output_state = output_state
            xywh_init_state = init_state
        elif self.state_type == "xcycwh":
            xywh_output_state = xcycwh_to_xywh(output_state)
            xywh_init_state = xcycwh_to_xywh(init_state)
        elif self.state_type == "xcycah":
            xywh_output_state = xcycah_to_xywh(output_state)
            xywh_init_state = xcycah_to_xywh(init_state)
        elif self.state_type == "x1y1ah":
            xywh_output_state = x1y1ah_to_xywh(output_state)
            xywh_init_state = x1y1ah_to_xywh(init_state)
        else:
            raise ValueError("unknown state type")
        # Then recover
        recovered_xywh_output_state = recover_xywh_bbox_pos(xywh_init_state, xywh_output_state)
        # Then convert back to the original format
        if self.state_type == "xywh":
            return recovered_xywh_output_state
        elif self.state_type == "xcycwh":
            return xywh_to_xcycwh(recovered_xywh_output_state)
        elif self.state_type == "xcycah":
            return xywh_to_xcycah(recovered_xywh_output_state)
        elif self.state_type == "x1y1ah":
            return xywh_to_x1y1ah(recovered_xywh_output_state)
        else:
            raise ValueError("unknown state type")

    def predict(self, new_state=None, update_history_bank=True, update_prediction_bank=False):
        """
        Use the trained model to predict the future trajectory.
        When a new state is given, update the historical trajectory bank and use the model to predict the next state.
        When no new state is given, only use the model to predict the next state.
        new_state: Tensor, shape (state_dim,).
        """
        if new_state is not None and update_history_bank:
            self.update_history_bank(new_state)
            history_states_list = self.history_bank.copy()
        elif new_state is not None and not update_history_bank:
            history_states_list = self.history_bank.copy()
            # Add new_state to the list according to the direction
            if not self.reverse:
                history_states_list.append(new_state)
                if len(history_states_list) > self.history_size:
                    history_states_list.pop(0)
            else:
                history_states_list.insert(0, new_state)
                if len(history_states_list) > self.history_size:
                    history_states_list.pop()
        else:
            history_states_list = self.history_bank.copy()

        original_init_state = history_states_list[0]
        
        if len(history_states_list) < self.history_size:
            return None  # Insufficient history, cannot predict
        if self.normalize_box_size:
            history_states_list = self._normalize_bbox_size(history_states_list)
        if self.normalize_box_pos:
            history_states_list = self._normalize_bbox_pos(history_states_list)
        history_states = torch.tensor(history_states_list).view(-1).float()
        backbone_pred = self.model(history_states).detach().tolist()
        final_pred = backbone_pred
        if self.normalize_box_size:
            final_pred = self._recover_bbox_size(original_init_state, final_pred)
        if self.normalize_box_pos:
            final_pred = self._recover_bbox_pos(original_init_state, final_pred)
        # Ensure the first four output values are non-negative
        for i in range(4):
            if final_pred[i] < 0:
                final_pred[i] = 0
        if update_prediction_bank:
            self._update_prediction_bank(backbone_pred)
        return final_pred
    
    def get_full_state(self, state_without_velocity):
        """
        Given a state without velocity, compute the new state's velocity from the last historical state and return the full state with velocity.
        state_without_velocity: Tensor, shape (state_dim,).
        """
        if len(self.history_bank) == 0:
            return state_without_velocity + [0 for _ in range(self.state_dim - 4)]
        velocity = [state_without_velocity[i] - self.history_bank[-1][i] for i in range(4)]
        return state_without_velocity + velocity
     
def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

def xywh_to_xyxy(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def xyxy_to_xcycwh(bbox):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [xc, yc, w, h]

def xcycwh_to_xyxy(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def xyxy_to_xcycah(bbox):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        h = 1
    a = w / h if h!= 0 else 0
    return [xc, yc, a, h]
    
def xyxy_to_x1y1ah(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        h = 1
    a = w / h if h!= 0 else 0
    return [x1, y1, a, h]

def xcycah_to_xyxy(bbox):
    xc, yc, a, h = bbox
    x1 = xc - a * h / 2
    y1 = yc - h / 2
    x2 = xc + a * h / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def x1y1ah_to_xyxy(bbox):
    x1, y1, a, h = bbox
    x2 = x1 + a * h
    y2 = y1 + h
    return [x1, y1, x2, y2]

def convert_bbox_to_xyxy(bbox, state_type):
    if state_type == "xywh":
        new_bbox = xywh_to_xyxy(bbox)
    elif state_type == "xcycwh":
        new_bbox = xcycwh_to_xyxy(bbox)
    elif state_type == "xcycah":
        new_bbox = xcycah_to_xyxy(bbox)
    elif state_type == "x1y1ah":
        new_bbox = x1y1ah_to_xyxy(bbox)
    else:
        raise ValueError("state_type should be 'xywh' or 'xcycah' or 'x1y1ah'")
    return new_bbox
    
def _compute_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bbox1 : list
        The first bounding box in the format [x1, y1, x2, y2].
    bbox2 : list
        The second bounding box in the format [x1, y1, x2, y2].
    Returns
    -------
    float
        The IoU of the two bounding boxes.
    """
    if bbox2 == [0, 0, 0, 0]:
        return 0
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    # Calculate intersection area
    intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
    # Calculate union area
    union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection_area
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def compute_iou(pred_bbox, bboxes, state_type, convert_bboxes_as_well=False):
    """
    Compute the IoU between the bbox and the bboxes
    """
    ious = []
    pred_bbox = convert_bbox_to_xyxy(pred_bbox, state_type)
    for bbox in bboxes:
        if convert_bboxes_as_well:
            bbox = convert_bbox_to_xyxy(bbox, state_type)
        iou = _compute_iou(pred_bbox, bbox)
        ious.append(iou)
    return ious

def _compute_shape_score(bbox1, bbox2):
    """
    compute the shape similarity score between two bounding boxes.
    bbox1: list, the first bounding box in the format [x1, y1, x2, y2].
    bbox2: list, the second bounding box in the format [x1, y1, x2, y2].
    return: float, the shape similarity score.
    """
    if bbox1[3] - bbox1[1] == 0 and bbox2[3] - bbox2[1] == 0:
        return 1
    elif bbox1[3] - bbox1[1] == 0 or bbox2[3] - bbox2[1] == 0:
        return 0
    else:
        shape1 = (bbox1[2] - bbox1[0]) / (bbox1[3] - bbox1[1])
        shape2 = (bbox2[2] - bbox2[0]) / (bbox2[3] - bbox2[1])
        # assert shape1 >= 0 and shape2 >= 0, f"shape1 ({shape1}) and shape2 ({shape2}) should be positive"
        if not (shape1 >= 0 and shape2 >= 0):
            return 0
        if shape1 == 0 and shape2 == 0:
            return 1
        else:
            return min(shape1, shape2) / max(shape1, shape2)

def compute_shape_score(pred_bbox, bboxes, state_type="xywh", convert_bboxes_as_well=False):
    """
    Compute the size similarity score between the pred bbox and the bboxes
    """
    scores = []
    pred_bbox = convert_bbox_to_xyxy(pred_bbox, state_type)
    for bbox in bboxes:
        if convert_bboxes_as_well:
            bbox = convert_bbox_to_xyxy(bbox, state_type)
        score = _compute_shape_score(pred_bbox, bbox)
        scores.append(score)
    return scores

def _compute_size_score(bbox1, bbox2):
    """
    compute the size similarity score between two bounding boxes.
    bbox1: list, the first bounding box in the format [x1, y1, x2, y2].
    bbox2: list, the second bounding box in the format [x1, y1, x2, y2].
    return: float, the size similarity score.
    """
    size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # assert size1 >= 0 and size2 >= 0, "size1 and size2 should be positive"
    if not (size1 >= 0 and size2 >= 0):
        return 0
    if size1 == 0 and size2 == 0:
        return 1
    else:
        return min(size1, size2) / max(size1, size2)

def compute_size_score(pred_bbox, bboxes, state_type="xywh", convert_bboxes_as_well=False):
    """
    Compute the size similarity score between the pred bbox and the bboxes
    """
    scores = []
    pred_bbox = convert_bbox_to_xyxy(pred_bbox, state_type)
    for bbox in bboxes:
        if convert_bboxes_as_well:
            bbox = convert_bbox_to_xyxy(bbox, state_type)
        score = _compute_size_score(pred_bbox, bbox)
        scores.append(score)
    return scores

def _compute_center_distance_score(bbox1, bbox2):
    """
    compute the center distance score between two bounding boxes.
    bbox1: list, the first bounding box in the format [x1, y1, x2, y2].
    bbox2: list, the second bounding box in the format [x1, y1, x2, y2].
    return: float, the center distance score.
    """
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    # Calculate the diagonal lengths of the bounding boxes
    diag1 = ((bbox1[2] - bbox1[0]) ** 2 + (bbox1[3] - bbox1[1]) ** 2) ** 0.5
    diag2 = ((bbox2[2] - bbox2[0]) ** 2 + (bbox2[3] - bbox2[1]) ** 2) ** 0.5
    
    if diag1 == 0:  # Handle edge cases where pred bounding box have zero size
        return 1.0 if distance == 0 else 0.0
    
    # Calculate the center distance score
    score = 1 - min(distance / diag1, 1)  # Clamp the score to [0, 1]
    return score

def compute_center_distance_score(pred_bbox, bboxes, state_type="xywh", convert_bboxes_as_well=False):
    """
    Compute the center distance score between the pred bbox and the bboxes
    """
    scores = []
    pred_bbox = convert_bbox_to_xyxy(pred_bbox, state_type)
    for bbox in bboxes:
        if convert_bboxes_as_well:
            bbox = convert_bbox_to_xyxy(bbox, state_type)
        score = _compute_center_distance_score(pred_bbox, bbox)
        scores.append(score)
    return scores
