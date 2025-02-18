from functools import partial
import os
import numpy as np
import torch
import pickle as pkl
from PIL import Image
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from common.quaternion import *

from utils.rotation_conversions import *
from utils.humanml_utils import HML_LOWER_BODY_JOINTS, SMPL_UPPER_BODY_JOINTS, SMPL_ARM_JOINTS

face_joint_indx = [2,1,17,16]
fid_l = [7,10]
fid_r = [8,11]

from tqdm import tqdm
mytqdm = partial(tqdm, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}')

def print_check_tensor(x, name, froce=False):
    if isinstance(x, torch.Tensor):
        if not torch.isnan(x).any():
            return
        # list index of nan values
        nan_idx = torch.isnan(x).nonzero(as_tuple=False)

        stat = [torch.norm(x), x.mean(), x.std(), x.abs().max(), 
                        torch.isnan(x).any(), torch.isinf(x).any()]
        print("T: ", name, " ".join([str(s.cpu().item()) for s in stat]), x.shape)
        print("nan idx", nan_idx)
        raise ValueError("Nan value detected")
    elif isinstance(x, np.ndarray):
        if not np.isnan(x).any():
            return
        # list index of nan values
        nan_idx = np.argwhere(np.isnan(x))

        stat = [np.linalg.norm(x), x.mean(), x.std(), np.abs(x).max(),
                        np.isnan(x).any(), np.isinf(x).any()]
        print("N: ", name, " ".join([str(s) for s in stat]), x.shape)
        print("nan idx", nan_idx)
        raise ValueError("Nan value detected")
    else:
        pass


def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def rigid_transform(relative, data):

    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data


class Normalizer():
    def __init__(self, load_file=None, stat=None, device="cpu"):
        """_summary_

        Args:
            stat (_type_, optional): _description_. Defaults to None.
            device (str, optional): _description_. Defaults to "cpu".
        """
        device = "cpu"
        if load_file is not None:
            print("[INFO] Load normalizer from ", load_file)
            with open(load_file, "rb") as f:
                stat = pkl.load(f)
        # DATA_ROOT = "./mydata/motions_processed"
        # if stat is None:
        # with open(os.path.join(DATA_ROOT, "data_statistics.pkl"), "rb") as f:
        #     stat = pkl.load(f)
        self.stat = stat.copy()
        self.stat_t = stat.copy()
        
        for k, v in self.stat.items():
            self.stat_t[k] = torch.from_numpy(v).float().to(device).reshape(1, -1)
            self.stat[k] = v.reshape(1, -1)
            
    def get_stat_key(self, name):
        if name is None:
            m, s = "mean", "std"
        else:
            m, s = name+"_mean", name+"_std"
        return m, s

    def forward(self, x, normal_slice=None, name=None):
        m, s = self.get_stat_key(name)
            
        if isinstance(x, np.ndarray):
            std, mean = self.stat[s], self.stat[m]
            if normal_slice is not None:
                std, mean = std[:, normal_slice], mean[:, normal_slice]
            x = (x - mean) / std
        else:
            std, mean = self.stat_t[s], self.stat_t[m]
            if normal_slice is not None:
                std, mean = std[:, normal_slice], mean[:, normal_slice]
            x = (x - mean.to(device=x.device)) / std.to(device=x.device)

        return x

    def backward(self, x, normal_slice=None, name=None):
        m, s = self.get_stat_key(name)
        
        if isinstance(x, np.ndarray):
            std, mean = self.stat[s], self.stat[m]
            if normal_slice is not None:
                std, mean = std[:, normal_slice], mean[:, normal_slice]
            x = x * std + mean
        else:
            std, mean = self.stat_t[s], self.stat_t[m]
            if normal_slice is not None:
                std, mean = std[:, normal_slice], mean[:, normal_slice]
            x = x * std.to(device=x.device) + mean.to(device=x.device)
        return x


class MotionNormalizer():
    def __init__(self, clip_upper=True):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")
        if clip_upper:
            self.motion_mean_full = mean
            self.motion_std_full = std
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
            mean = extract_upper_motions(mean)
            std = extract_upper_motions(std)

        self.clip_upper = clip_upper

        self.motion_mean = mean
        self.motion_std = std


    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x

    def backward(self, x, full=False):
        if full and self.clip_upper:
            x = x * self.motion_std_full + self.motion_mean_full
        else:
            x = x * self.motion_std + self.motion_mean
        return x



class MotionNormalizerTorch():
    def __init__(self, clip_upper=True):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")
        if clip_upper:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
            mean = extract_upper_motions(mean)
            std = extract_upper_motions(std)

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()

    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x

trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def process_motion_np(motion, feet_thre, prev_frames, n_joints, fix_root=True):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints*3:]

    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    '''XZ at origin'''
    if fix_root:
        root_pos_init = positions
        root_pose_init_xz = root_pos_init[:, 0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz[:, None]
        root_pose_init_xz = root_pose_init_xz[0]
    else:
        root_pos_init = positions[prev_frames]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz
        # print("root_pose_init_xz", root_pose_init_xz.shape)
    
    print_check_tensor(positions, "positions after pos")
        

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    if fix_root:
        across = root_pos_init[:, r_hip] - root_pos_init[:, l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]]).repeat(forward_init.shape[0], axis=0)
        print_check_tensor(forward_init, "forward_init")
        root_quat_init = qbetween_np(forward_init, target)
        print_check_tensor(root_quat_init, "root_quat_init")
        root_quat_init_for_all = root_quat_init.reshape(positions.shape[0], 1, 4).repeat(positions.shape[1], axis=1)
        root_quat_init = root_quat_init[:1]
    else:
        across = root_pos_init[r_hip] - root_pos_init[l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
        # print("root_quat_init", root_quat_init.shape)


    positions = qrot_np(root_quat_init_for_all, positions)
    print_check_tensor(positions, "positions after qrot")

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    rot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]


def extract_upper_motions(motions):
    # motion shape: (seq_len, joints*3 + joint_vels*3 + rotations*6 + feet_l+feet_e)
    n_joints = 22
    n_rots = 21
    motions_pos = motions[:, :n_joints*3].reshape(-1, n_joints, 3)
    motions_vel = motions[:, n_joints*3:n_joints*6].reshape(-1, n_joints, 3)
    motions_rot = motions[:, n_joints*6:n_joints*6+n_rots*6].reshape(-1, n_rots, 6)
    upper_joints = [0] + SMPL_UPPER_BODY_JOINTS
    motions_pos = motions_pos[:, upper_joints].reshape(-1, len(upper_joints)*3)
    motions_vel = motions_vel[:, upper_joints].reshape(-1, len(upper_joints)*3)
    motions_rot = motions_rot[:, np.array(SMPL_UPPER_BODY_JOINTS, dtype=np.int64)-1].reshape(-1, len(SMPL_UPPER_BODY_JOINTS)*6)
    return np.concatenate([motions_pos, motions_vel, motions_rot], axis=-1)


def extract_upper_motions_simple(motions, feat_unit_dim=6):
    """ 
        whole body to two arms: 24->8 
        original dim: 24*[pos(3), axis-angle(3), quat(4), 6d(6)]
    """
    n_rots = len(SMPL_ARM_JOINTS)  # 8
    # motions_rot = motions[:, n_joints*6:n_joints*6+n_rots*6].reshape(-1, n_rots, 6)
    motions = motions.reshape(-1, 24, feat_unit_dim)
    motions_rot = motions[:, np.array(SMPL_ARM_JOINTS, dtype=np.int64)].reshape(-1, n_rots*feat_unit_dim)
    return motions_rot


def recover_upper2full(upper_motions, ref_motion=None):
    n_joints = 22
    n_rots = 21
    n_ujoints = 14
    n_urots = 13
    B = upper_motions.shape[0]
    motions = np.zeros((upper_motions.shape[0], n_joints*2*3+n_rots*6+4))
    if ref_motion is None:
        ref_motion = np.zeros((1, n_joints*2*3+n_rots*6+4))
    motions[:] = ref_motion

    motions_pos = motions[:, :n_joints*3].reshape(-1, n_joints, 3)
    motions_vel = motions[:, n_joints*3:n_joints*6].reshape(-1, n_joints, 3)
    motions_rot = motions[:, n_joints*6:n_joints*6+n_rots*6].reshape(-1, n_rots, 6)

    upper_joints = [0] + SMPL_UPPER_BODY_JOINTS
    upper_rots = np.array(SMPL_UPPER_BODY_JOINTS)-1
    motions_pos[:, upper_joints] = upper_motions[:, :n_ujoints*3].reshape(B, -1, 3)
    motions_vel[:, upper_joints] = upper_motions[:, n_ujoints*3:n_ujoints*6].reshape(B, -1, 3)
    motions_rot[:, upper_rots] = upper_motions[:, n_ujoints*6:n_ujoints*6+n_urots*6].reshape(B, -1, 6)
    
    motions_pos = motions_pos.reshape(-1, n_joints*3)
    motions_vel = motions_vel.reshape(-1, n_joints*3)
    motions_rot = motions_rot.reshape(-1, n_rots*6)
    foot = np.zeros((B, 4))
    motions = np.concatenate([motions_pos, motions_vel, motions_rot, foot], axis=-1)
    return motions

def recover_upper2smpl_rot(upper_motions):
    """
    Args: upper_motions: (B, 13, 6)
    Returns: (B, 23, 6) no global orientation
    """
    n_rots = 23
    B = upper_motions.shape[0]
    motions = torch.zeros((B, n_rots, 6), dtype=upper_motions.dtype, device=upper_motions.device)
    upper_rots = torch.tensor(SMPL_UPPER_BODY_JOINTS, device=upper_motions.device)-1
    motions[:, upper_rots] = upper_motions
    return motions

def extract_upper_from_smpl(smpl_pos):
    """
    Args: smpl_pos: (B, 24, 3)
    Returns: motion_pos (B, 14, 3)
    """
    upper_joints = torch.tensor([0] + SMPL_UPPER_BODY_JOINTS, device=smpl_pos.device)
    return smpl_pos[:, upper_joints]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

