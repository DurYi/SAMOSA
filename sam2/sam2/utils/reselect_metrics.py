import numpy as np

def _compute_centroid(box):
    """Compute the center point of a box."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def _compute_velocity(box_list):
    """Compute centroid velocities."""
    centroids = np.array([_compute_centroid(box) for box in box_list])
    velocities = np.diff(centroids, axis=0)
    return velocities

def _compute_acceleration(velocities):
    """Compute accelerations."""
    accelerations = np.diff(velocities, axis=0)
    return accelerations

def _acceleration_rmse(box_list):
    """Compute the acceleration root mean square error (Acceleration RMSE)."""
    velocities = _compute_velocity(box_list)
    accelerations = _compute_acceleration(velocities)
    if len(accelerations) == 0:
        return 0
    rmse = np.sqrt(np.mean(np.sum(accelerations ** 2, axis=1)))
    return rmse

def compute_acceleration_rmse(bbox_list, pred_bboxes, normalize=True, lambda_=0.05):
    """
    Compute the acceleration RMSE between the pred bbox and the bboxes
    """
    scores = []
    for pred_bbox in pred_bboxes:
        full_list = bbox_list + [pred_bbox]
        score = _acceleration_rmse(full_list)
        if normalize:
            score = 1-np.exp(-score*lambda_)
        scores.append(score)
    return scores
