import glob
import h5py
import numpy as np
import pyzed.sl as sl
import time
import cv2
import sys
import matplotlib.pyplot as plt 
import tqdm
import torch
from torch.utils.data import Dataset
import os 
import sys
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from collections import Counter
import yaml

from pytransform3d import rotations
import concurrent.futures
from pathlib import Path
import argparse
import pickle as pkl

sys.path.append(os.curdir)

kPi_2 = 1.57079632
RIGHT_ARM_QPOS = np.array([0.0, 0.0, 1.3, kPi_2, 0])
RIGHT_HAND_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def split_indices(labels, task=[0, 1, 2, 3], prelong=0, prelong_only=None):
    slices = {}
    current_slice = {}
    for t in task:
        slices[t] = []
        current_slice[t] = []
    current = labels[0]
    for i, label in enumerate(labels):
        if current != label:
            if current in task:
                slices[current].append(current_slice[current])
                current_slice[current] = []
            if label in task:
                if prelong_only is None or label == prelong_only:
                    current_slice[label] = list(range(i-prelong, i+1)) if i-prelong >= 0 else list(range(0, i+1))
                else:
                    current_slice[label] = [i]
        else:
            if label in task:
                current_slice[label].append(i)
        current = label
        
    for t in task:
        if len(current_slice[t]) > 15:
            # add prelong
            if prelong_only is None or t == prelong_only:
                tmp = current_slice[t].copy()
                current_slice[t] = list(range(tmp[0]-prelong, tmp[0]))
                current_slice[t].extend(tmp)
            slices[t].append(current_slice[t])
                
    return slices

def load_svo(path, crop_size_h=240, crop_size_w=320, with_label=False, no_crop=False, just_resize=False):
    input_file = path + "/zed.svo2"
    # import ipdb; ipdb.set_trace()
    # print(input_file)
    crop_size_h = crop_size_h
    crop_size_w = crop_size_w
    if no_crop:
        crop_size_h = 0
        crop_size_w = 0
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(input_file)
    print("loading svo", input_file)
    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(-1)
    left_image = sl.Mat()
    right_image = sl.Mat()

    depth_image = sl.Mat()

    nb_frames = zed.get_svo_number_of_frames()
    print("Total image frames: ", nb_frames)

    cropped_img_shape = (720-crop_size_h, 1280-2*crop_size_w)
    left_imgs = np.zeros((nb_frames, 3, cropped_img_shape[0], cropped_img_shape[1]), dtype=np.uint8)
    right_imgs = np.zeros((nb_frames, 3, cropped_img_shape[0], cropped_img_shape[1]), dtype=np.uint8)
    timestamps = np.zeros((nb_frames, ), dtype=np.int64)
    cnt = 0
    tar_size = (640, 480)
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            timestamps[cnt] = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            
            if just_resize:
                left_imgs[cnt] = cv2.cvtColor(cv2.resize(left_image.get_data(), tar_size), cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
                right_imgs[cnt] = cv2.cvtColor(cv2.resize(right_image.get_data(), tar_size), cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)   
            elif no_crop:
                left_imgs[cnt] = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
                right_imgs[cnt] = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
            else:
                left_imgs[cnt] = cv2.cvtColor(left_image.get_data()[crop_size_h:, crop_size_w:-crop_size_w], cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
                right_imgs[cnt] = cv2.cvtColor(right_image.get_data()[crop_size_h:, crop_size_w:-crop_size_w], cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
            cnt += 1
            
        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
    
    labels = None
    if with_label:
        label_file = path + "/videos/label.pkl"
        labels = np.load(label_file, allow_pickle=True)
    return left_imgs, right_imgs, timestamps, labels

def check_states(states):
    left_elbow = states[:, 5]
    plt.figure()
    plt.plot(left_elbow, label='new')
    plt.legend()
    plt.savefig('elbow_states.png')


def check_actions(actions, left=True):
    old_actions = actions.copy()
    if left:
        fname = "motor_logs/left_elbow_bound/state_logs.pkl"
    else:
        fname = "motor_logs/right_elbow_bound/state_logs.pkl"
    with open(fname, "rb") as f:
        data = pkl.load(f)
    ArmJointsName = [
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
    ]
    # xname = ["left_elbow_q", "left_shoulder_q", "left_shoulder_yaw_q", "left_shoulder_pitch_q", "left_shoulder_roll_q", "left_elbow_yaw_q", "left_elbow_pitch_q"]
    if left:
        left_xname = [ArmJointsName[i]+"_q" for i in [0,1,2]]
        yvalue = data[ArmJointsName[3]+"_q"]
    else:
        left_xname = [ArmJointsName[i]+"_q" for i in [4,5,6]]
        yvalue = data[ArmJointsName[7]+"_q"]
    for action in actions:
        if left:
            pos = action[2:5]
            val = action[5]
        else:
            pos = action[6:9]
            val = action[9]
        threshold = 0.3
        min_c = 10
        closest = 1
        closet_c = 0
        for i in range(len(data[left_xname[0]])):
            x, y, z = data[left_xname[0]][i], data[left_xname[1]][i], data[left_xname[2]][i]
            c = yvalue[i]
            dist = np.linalg.norm(pos - np.array([x, y, z]))
            if dist < closest:
                closest = dist
                closest_c = c
            if dist < threshold:
                if c < min_c:
                    min_c = c
        if min_c == 10:
            print(f'closest {closest_c} clipped to {min_c}')
            min_c = closest_c

        if val > min_c:
            # print(f'elbow pos {val} clipped to {max_c}')
            if left:
                action[5] = min_c
            else:
                action[9] = min_c

    return actions


def load_hdf5(path):  # offset 10ms
    input_file = path + "/data.hdf5"
    file = h5py.File(input_file, 'r')
    
    timestamps = np.array(file["timestamp"][:], dtype=np.int64) 
    video_start = np.array(file["video_start_stamp"], dtype=np.int64)
    # timestamps = np.array(file["timestamp"][:] * 1000, dtype=np.int64)
    states = np.array(file["h1_state"][:])  # (timestamps.shape[0], 37=2+8+2+12+12+1)
    actions = np.array(file["cmd_pos"][:])  # (timestamps.shape[0], 36=2+10+24)

    return timestamps, states, actions, video_start

def match_timestamps(candidate, ref):
    closest_indices = []
    # candidate = np.sort(candidate)
    for t in ref:
        idx = np.searchsorted(candidate, t, side="left")
        if idx > 0 and (idx == len(candidate) or np.fabs(t - candidate[idx-1]) < np.fabs(t - candidate[idx])):
            closest_indices.append(idx-1)
        else:
            closest_indices.append(idx)
    # print("closest_indices: ", len(closest_indices))
    return np.array(closest_indices)

def process_head_action(head):
    mat = head.reshape(3, 3)
    if np.sum(mat) == 0:
        mat = np.eye(3)
    head_rot = rotations.quaternion_from_matrix(mat[:3, :3])
    ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
    return ypr[:2]

def process_hand_action(hand):
    '''
    hand joints:
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint", (mimic)
    "L_thumb_distal_joint",(mimic)
    "L_index_proximal_joint",
    "L_index_intermediate_joint",(mimic)
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",(mimic)
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",(mimic)
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",(mimic)
    '''
    assert hand.shape == (24, )
    return hand[[0,1,4,6,8,10, 12,13,16,18,20,22]]

def process_force(force):
    """
    receive: right, left
    processed: left, right
    """
    assert force.shape == (12, )
    right, left = force[:6], force[6:]
    
    return np.concatenate([left, right])

def process_hand_obs(hand):
    """
    receive: right, left
    processed: left, right 
    (! changed left and right!!!)
    """
    assert hand.shape == (12, )
    right, left = hand[:6], hand[6:]
    return np.concatenate([left, right])

def save_video(left_imgs, path, fps=60):
    _, height, width= left_imgs[0].shape
    print(f"width: {width}, height: {height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for img in left_imgs:
        # print(img.shape)
        img_bgr = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)

    video_writer.release()

def match_timestamps(hdf5_timestamps, video_start, video_len):
    indices = []
    data_start = -1
    data_end = len(hdf5_timestamps)
    for i, time in enumerate(hdf5_timestamps):
        if time >= video_start and time - video_start < video_len:
            if data_start < 0:
                data_start = i
            indices.append(time - video_start)
        if time - video_start >= video_len:
            data_end = i
            break

    return data_start, data_end, indices
    # data = data[data_start:data_end]
    # imgs = imgs[indices]

def process_episode(dir_name, only_video=False, with_label=False, slow_video=False, only_action=False, react=False, just_resize=False,config={}):
    print("=====================================")
    if only_action:
        hdf5_timestamps, states, actions, video_start = load_hdf5(dir_name)
        return
    
    left_imgs, right_imgs, img_timestamps, labels = load_svo(dir_name, with_label=with_label, no_crop=react, just_resize=just_resize)

    hdf5_timestamps, states, actions, video_start = load_hdf5(dir_name)

    data_start, data_end, indices = match_timestamps(hdf5_timestamps, video_start, img_timestamps.shape[0])
    left_imgs = left_imgs[indices]
    right_imgs = right_imgs[indices]
    if with_label:
        if not labels.shape[0] == left_imgs.shape[0]:
            print("original labels: ", labels.shape)
            labels = labels[indices]
    
    states = states[data_start:data_end]
    actions = actions[data_start:data_end]
    print("left_imgs: ", left_imgs.shape)
    print("actions: ", actions.shape)
    
    os.makedirs(os.path.join(dir_name, "videos"), exist_ok=True)
    if slow_video:
        save_video(left_imgs, os.path.join(dir_name, "videos", "sample_slow.mp4"), fps=30)
    else:
        save_video(left_imgs, os.path.join(dir_name, "videos", "sample_left.mp4"))
        save_video(right_imgs, os.path.join(dir_name, "videos", "sample_right.mp4"))
    if only_video:
        return

    # process actions
    head, arm, hand = actions[:, :2], actions[:, 2:12], actions[:, -24:]  # 2+10+24
    # head = np.stack([process_head_action(h) for h in head], axis=0)  # (timesteps, 2)
    hand = np.stack([process_hand_action(h) for h in hand], axis=0)  # (timesteps, 12)
    qpos_actions = np.concatenate([head, arm, hand], axis=1)  # (timesteps, 24=2+10+12)
    print("qpos_actions: ", qpos_actions.shape)
    assert qpos_actions.shape[1] == 24
    assert left_imgs.shape[0] == qpos_actions.shape[0]

    # process states
    if not react:
        head_obs, arm_obs, hand_obs, forces = states[:, :2], states[:, 2:12], states[:, 12:24], states[:, 24:36]
        if states.shape[1] > 37:
            print("new arm states")
            arm_obs = np.concatenate([states[:, 36:44], states[:, 10:12]], axis=-1)
        forces = np.stack([process_force(f) for f in forces], axis=0)  # (timesteps, 12)
        hand_obs = np.stack([process_hand_obs(h) for h in hand_obs], axis=0)  # (timesteps, 12)
        states = np.concatenate([head_obs, arm_obs, hand_obs, forces], axis=1)  # (timesteps, 36=2+10+12+12)
        
        print("states: ", states.shape)
        assert states.shape[1] == 36
        assert left_imgs.shape[0] == states.shape[0] == qpos_actions.shape[0]
    

    if with_label:
        if len(labels.shape) == 2:
            skill_labels = labels[:, 0]
            unsafe_labels = labels[:, 1]
        else:
            skill_labels = labels
            unsafe_labels = None
            
        
        task = config['labels']
        print('Label statistics:', Counter(skill_labels))
        slices = split_indices(skill_labels, task=task.keys(), prelong=config['prelong'], prelong_only=config['prelong_only'])
    
    if with_label:
        for id, name in task.items():
            os.makedirs(os.path.join(dir_name, f"processed_{name}"), exist_ok=True)
            for i, slice in enumerate(slices[id]):
                with h5py.File(dir_name + f"/processed_{name}/processed_{i}.hdf5", 'w') as hf:
                    hf.create_dataset('observation.image.left', data=left_imgs[slice], )
                    hf.create_dataset('observation.image.right', data=right_imgs[slice], )
                    if not react:
                        hf.create_dataset('observation.state', data=states[slice].astype(np.float32), )
                    hf.create_dataset('qpos_action', data=qpos_actions[slice].astype(np.float32), )
                    if unsafe_labels is not None:
                        hf.create_dataset('cancel_unsafe', data=unsafe_labels[slice].astype(np.float32), )
                    hf.attrs['sim'] = False
    
    else:
        with h5py.File(dir_name + f"/processed.hdf5", 'w') as hf:
            hf.create_dataset('observation.image.left', data=left_imgs, )
            hf.create_dataset('observation.image.right', data=right_imgs, )
            if not react:
                hf.create_dataset('observation.state', data=states.astype(np.float32), )
            hf.create_dataset('qpos_action', data=qpos_actions.astype(np.float32), )
            hf.attrs['sim'] = False
     

def find_all_episodes(path):
    episodes = [os.path.join(path, f) for f in os.listdir(path)]
    print(episodes)
    episodes = [os.path.basename(ep).split(".")[0] for ep in episodes]
    print(episodes)
    return episodes

def start_process_1episode(root, task, ep, only_video, with_label, slow_video=False, only_action=False, react=False, just_resize=False, config={}):
    dir_name = os.path.join(root, task)
    ep_name = dir_name + "/" + ep
    processed_files = glob.glob(ep_name + "/processed*")
    if args.overwrite or not processed_files:
        process_episode(ep_name, only_video, with_label, slow_video=slow_video, only_action=only_action, react=react, just_resize=just_resize,config=config)
        print('processed ep', ep_name)

def start_process_task(root, task, only_video, with_label=False, react=False, just_resize=False, config={}):
    dir_name = os.path.join(root, task)
    print(dir_name)
    all_eps = find_all_episodes(dir_name)
    for ep in all_eps:
        start_process_1episode(root, task, ep, only_video, with_label, react=react, just_resize=just_resize, config=config)

def load_process_config_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    default_config = config["DefaultCfg"]
    process_config = {}
    
    for dataset in config['DatasetCfg']:
        dataset_config = default_config.copy()
        dataset_config.update(dataset)
        
        process_config[dataset_config["name"]] = dataset_config


    return process_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="teleop_data_long_share")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--ep', type=str, required=False)
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--only-video', action="store_true")
    parser.add_argument('--slow-video', action="store_true")
    parser.add_argument('--only-action', action="store_true")
    parser.add_argument('--with-label', action="store_true")
    parser.add_argument('--react', action="store_true")  # no crop and no state
    parser.add_argument('--just-resize', action="store_true")
    parser.add_argument('--config', type=str, default="../dataset/process_config.yaml")
    args = parser.parse_args()

    """
    e.g. 
    python scripts/post_process_teleop_data.py --root teleop_data_long/ --task pick_can_R --only-video
    python scripts/post_process_teleop_data.py --root teleop_data_long/ --task pick_can_R --with-label
    """
    
    process_config = load_process_config_from_yaml(args.config)

    if args.task is None:
        all_tasks = [f for f in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, f))]
        print(f"All tasks: {all_tasks}")
        # exit()
        for task in all_tasks:
            print(f"==================== Task: {task} ====================")
            start_process_task(
                args.root, 
                task, 
                args.only_video,
                args.with_label, 
                args.react, 
                args.just_resize,
                process_config[task]
            )
    elif args.ep:
        start_process_1episode(
            args.root, 
            args.task, 
            args.ep, 
            args.only_video, 
            args.with_label, 
            args.slow_video, 
            args.only_action, 
            args.react, 
            args.just_resize,
            process_config[args.task]
        )
    else:
        start_process_task(
            args.root, 
            args.task, 
            args.only_video, 
            args.with_label, 
            args.react, 
            args.just_resize,
            process_config[args.task]
        )
