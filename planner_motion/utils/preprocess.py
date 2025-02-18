import numpy as np
import h5py
from utils.utils import *

FPS = 30

def load_motion(file_path, min_length, swap=False, reverse=False):
    """  """

    try:
        motion = np.load(file_path).astype(np.float32)
    except:
        print("error: ", file_path)
        return None, None
    motion1 = motion[:, :22 * 3]
    motion2 = motion[:, 62 * 3:62 * 3 + 21 * 6]
    motion = np.concatenate([motion1, motion2], axis=1)
    
    if reverse:
        motion = np.concatenate([motion, motion[::-1][1:]], axis=0)

    if motion.shape[0] < min_length:
        return None, None
    if swap:
        motion_swap = swap_left_right(motion, 22)
    else:
        motion_swap = None

    return motion, motion_swap

def load_motion_h1(file_path, file_path_vr=None):
    """
        h1 motion data:
        two arms: 2*5
    """
    motion = np.load(file_path, allow_pickle=True).astype(np.float32)
    if file_path_vr is None:
        return motion
    
    vr_motion = np.load(file_path_vr, allow_pickle=True).astype(np.float32)
    assert vr_motion.shape == motion.shape == (len(motion), 2*5), f"{vr_motion.shape} != {motion.shape} != {(len(motion), 2*5)}"
    motion[:, 4] = vr_motion[:, 4]
    motion[:, 9] = vr_motion[:, 9]

    return motion

def load_motion_human(file_path,):
    """
        human motion data: (T, 6, 6) ( from zed_human_rots )
    """
    data = np.load(file_path, allow_pickle=True).astype(np.float32)
    motion = data.reshape(-1, 6*6)
    # motion = motion[::2]
    return motion

def load_motion_simple(file_path, min_length, feat_types=None, swap=False, reverse=False):
    """
        pos, axis, qua, 6drot
        24*3 + 24*3 + 24*4 + 24*6
        default use 6drot
    """
    if feat_types is None:
        feat_types = ["6d"]
    n_joints = 24
    feats = ["pos", "axis", "qua", "6d"]
    dims = [3,3,4,6]
    start_i = [sum(dims[:i])*n_joints for i in range(len(dims)+1)]
    
    try:
        motion = np.load(file_path).astype(np.float32)
    except Exception as e:
        print("error: ", file_path)
        raise e
        return None, None
    motions = []
    for i in range(len(dims)):
        if feats[i] in feat_types:
            start = start_i[i]
            end = start_i[i+1]
            motions.append(motion[:, start:end])
    
    # pos = motion[:, :24 * 3]
    # rot6d = motion[:, -24 * 6:]
    omo = motion
    motion = np.concatenate(motions, axis=1)

    return motion, None

def load_hand_simple(file_path, add_occu, hand_update, num_obj, hand_update_interval=0):
    """
        in: left 12 + right 12 (+ leftobj + rightobj)
        out: left (6+num_obj) + right (6+num_obj)
    """
    motion = np.load(file_path).astype(np.float32)
    
    assert hand_update_interval == 0, f"hand_update_interval {hand_update_interval} != 0"
    if hand_update_interval > 0:
        motion_new = []
        for t in range(motion.shape[0]):
            if t % hand_update == 0:
                last_motion = motion[t]
            motion_new.append(last_motion)
        motion = np.stack(motion_new, axis=0)
            
    if num_obj > 0:
        if motion.shape[1] == 24:
            motion = np.concatenate([motion, np.zeros((motion.shape[0], 2))], axis=1)
        assert motion.shape[1] == 12 + 12 + 2, f"hand motion shape {motion.shape}, file {file_path}"
        
        obj_l = np.bincount(motion[:, 24].astype(int)).argmax()
        obj_r = np.bincount(motion[:, 25].astype(int)).argmax()
        
        # change labels to one-hot with length num_obj
        left_labels, right_labels = np.zeros((motion.shape[0], num_obj)), np.zeros((motion.shape[0], num_obj))
        for i in range(motion.shape[0]):
            if motion[i, 24] > 0:
                if motion[i, 24] > num_obj:
                    print(file_path)
                    print("==================================")
                left_labels[i, int(motion[i, 24]-1)] = 1
            if motion[i, 25] > 0:
                right_labels[i, int(motion[i, 25]-1)] = 1
        motion = motion[:, [0,1,4,6,8,10, 12,13,16,18,20,22]]
        if add_occu:
            motion = np.concatenate([motion[:, :6], left_labels, motion[:, 6:], right_labels], axis=1)
        # print(np.sum(left_labels, axis=0), np.sum(right_labels, axis=0))
        return motion, obj_l, obj_r

def preprocess_hand_diou(hand_near, hand_iou_mean_pool=False, dis_thresh=.2):
    """better hand_near design

    Args:
        hand_near (np.ndarray): (..., (NUM_OBJ+4)*2)
        dis_thresh (float, optional): threshold for mask obj. Defaults to .2.
    """
    hand_near = hand_near.reshape(hand_near.shape[0], 2, -1)

    near_obj_mask = hand_near[..., -4:-3] > dis_thresh
    
    # 0. randomly mask near obj
    # if self.obj_mask_ratio > 0 and self.opt.MODE == "train" and \
    #     np.random.rand() > self.obj_mask_ratio:
    #     near_obj_mask[:] = 1
        
    empty_value = np.zeros_like(hand_near)
    empty_value[..., -4] = 10.
    
    # better state design
    
    # 1. ignore obj when distance > 0.2
    hand_near = np.where(near_obj_mask, empty_value, hand_near)
    
    hand_obj = hand_near[..., :-4]
    hand_diou = hand_near[..., -4:]
    
    # 2. max distance 10 -> 1
    hand_diou[..., 0] = np.clip(hand_diou[..., 0], 0., 1.)
    
    # 3. using max or mean iou
    max_iou = np.max(hand_diou[..., 1:], axis=-1, keepdims=True)
    mean_iou = np.mean(hand_diou[..., 1:], axis=-1, keepdims=True)
    
    if hand_iou_mean_pool:
        hand_diou = np.concatenate([hand_diou[..., :1], mean_iou], axis=-1)
    else:
        hand_diou = np.concatenate([hand_diou[..., :1], max_iou], axis=-1)
    hand_near = np.concatenate([hand_obj, hand_diou], axis=-1)
    
    hand_near = hand_near.reshape(hand_near.shape[0], -1)
    return hand_near

def preprocess_hand_pos(hand_pos_raw, add_z=False):
    """ ignore z-axis(herizontal) for hand_pos and add missing mask"""
    hand_pos = np.copy(hand_pos_raw)
    hand_pos = hand_pos.reshape(hand_pos.shape[0], -1)
    
    if not add_z:
        hand_pos = hand_pos[:, [0,1,3,4]]
    hand_mis = np.isnan(hand_pos[:,[0,3]]).astype(np.float32)
    hand_pos[np.isnan(hand_pos)] = 0.
    hand_pos = np.concatenate([hand_pos, hand_mis], axis=-1).astype(np.float32)
    return hand_pos

def preprocess_head_pos(head_pos_raw):
    head_pos = head_pos_raw  # (T, 3)
    
    head_pos = head_pos[:, [2]]
    # head_mis = np.isnan(head_pos[:, 2]).astype(np.float32)
    head_pos[np.isnan(head_pos)] = 0.
    # head_pos = np.concatenate([head_pos, head_mis], axis=-1).astype(np.float32)
    return head_pos

def parse_diou_1hot(hand_near_diou_raw, num_obj):
    """parse diou to 1hot

    Args:
        hand_near_diou_raw (np.ndarray): (..., 2+num_obj)
        num_obj (int): number of objects

    Returns:
        np.ndarray: (..., num_obj+1)
    """
    # parse the first id of near, id 2 onehot
    hand_near_diou = hand_near_diou_raw
    cid = np.zeros((hand_near_diou.shape[0]*2, num_obj+1), dtype=np.float32)
    c = hand_near_diou[..., 0].astype(int).reshape(-1)
    cid[np.arange(c.shape[0]), c] = 1
    # remove the c==-1
    cid = cid[:, :-1]
    
    # ["bottle", "cup", "bowl", "cake"],
    # ->["", "can", "cup", "plate", "tissue", "sponge"]
    
    cid = cid.reshape(-1, 2, num_obj)
    
    hand_near_diou = np.concatenate([cid, hand_near_diou[..., 1:]], axis=-1)
    
    hand_near_diou = hand_near_diou.reshape((hand_near_diou.shape[0], -1))
    return hand_near_diou

def load_hand_details(file_path, num_obj, 
                      ignore_img=True,
                      hand_pos_add_z=False,):
    """load hand details from hdf5 file
        [('hand', (2, 1050, 3)), ('imgs', (1050, 256, 128, 3)), ('near_diou', (2, 1050, 5)), ('objs', (2, 1050, 4, 4))]
    """
    with h5py.File(file_path, "r") as f:
        hand = f["hand"][:]
        if ignore_img:
            res = 1
            hand_imgs = np.zeros((hand.shape[0], 2, res, res, 3), dtype=np.uint8)
        else:
            hand_imgs = f["imgs"][:]
        hand_near_diou = f["near_diou"][:]
        hand_objs = f["objs"][:]
        if "head" in f:
            head = f["head"][:]
        else:
            # print("!!!!!!!!!! no head !!!!!!!!!")
            head = np.zeros((hand.shape[0], 3))
    
    hand = preprocess_hand_pos(hand, hand_pos_add_z)
    hand_near_diou = parse_diou_1hot(hand_near_diou, num_obj)
    head = preprocess_head_pos(head)  # (T, 1)
    
    # (-1, 2, 128, 128, 3)->(-1, 2, 3, 128, 128)
    # BGR -> RGB
    hand_imgs = hand_imgs[..., [2,1,0]]
    hand_imgs = np.transpose(hand_imgs, (0, 1, 4, 2, 3))
    hand_objs = hand_objs.reshape((hand.shape[0], -1))
    
    ret = [hand, hand_imgs, hand_near_diou, hand_objs, head]
    ret = [r.astype(np.float32) for r in ret]
    
    return ret