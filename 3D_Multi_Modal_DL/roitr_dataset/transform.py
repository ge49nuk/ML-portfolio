import numpy as np
import scipy.ndimage
import scipy.interpolate

"""
script modified from: https://github.com/3dlg-hcvc/minsu3d
"""


def jitter(intensity=0.1):
    """
    params:
        the intensity of jittering
    return:
        3x3 jitter matrix
    """
    return np.eye(3,) + np.random.randn(3, 3) * intensity


def flip(axis=0, random=False):
    """
    flip the specified axis
    params:
        axis 0:x, 1:y, 2:z
    return:
        3x3 flip matrix
    """
    m = np.eye(3)
    if axis == -1:
        axis = np.random.randint(0,3)
    m[axis][axis] *= -1 if not random else np.random.randint(0, 2) * 2 - 1
    return m


def rot():
    """
    Generate a random rotation matrix around all axes.
    """
    theta_x = np.random.uniform(0, 2*np.pi)
    theta_y = np.random.uniform(0, 2*np.pi)
    theta_z = np.random.uniform(0, 2*np.pi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    return rotation_matrix


def elastic(x, gran, mag):
    """
    Refers to https://github.com/Jia-Research-Lab/PointGroup/blob/master/data/scannetv2_inst.py
    """
    blur0 = np.ones((3, 1, 1), dtype=np.float32) / 3
    blur1 = np.ones((1, 3, 1), dtype=np.float32) / 3
    blur2 = np.ones((1, 1, 3), dtype=np.float32) / 3

    bb = (np.abs(x).max(0) // gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype(np.float32) for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    return x + np.hstack([i(x)[:, None] for i in interp]) * mag


def crop(pc, max_num_point, scale):
    """
    Crop the points such that there are at most max_num_points points
    """
    pc_offset = pc.copy()
    valid_idxs = pc_offset.min(1) >= 0
    max_pc_range = np.full(shape=3, fill_value=scale, dtype=np.uint16)
    pc_range = pc.max(0) - pc.min(0)
    while np.count_nonzero(valid_idxs) > max_num_point:
        offset = np.clip(max_pc_range - pc_range + 0.001, None, 0) * np.random.rand(3)
        pc_offset = pc + offset
        valid_idxs = np.logical_and(pc_offset.min(1) >= 0, np.all(pc_offset < max_pc_range, axis=1))
        max_pc_range[:2] -= 32
    return pc_offset, valid_idxs

def flip_bbox(cfg, bbox, axis=0):
    """
    flip bbox on specified axis
    params:
        axis 0: horizontally
        axis 1: vertically
    """
    bbox_mean, bbox_std = cfg["bbox_mean_std"]
    bbox = bbox * bbox_std + bbox_mean
    if axis == 0:
        bbox[0] = 223. - bbox[0] - bbox[2]
    else:
        bbox[1] = 223. - bbox[1] - bbox[3]
    bbox = (bbox - bbox_mean) / bbox_std
    return np.array(bbox, dtype=np.float32)