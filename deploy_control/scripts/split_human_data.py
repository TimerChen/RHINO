import os
import shutil
import numpy as np
import h5py
import pickle as pkl
import argparse
import sys
import cv2

sys.path.append(os.curdir)

def hand6to12(hand_pos):
    assert(len(hand_pos) == 6)
    target_pos = np.array([hand_pos[0], hand_pos[1], hand_pos[1] * 1.4, hand_pos[1] * 0.6, 
                hand_pos[2], hand_pos[2], hand_pos[3], hand_pos[3], hand_pos[4], hand_pos[4], hand_pos[5], hand_pos[5]])
    return target_pos


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

def merge_indices(labels, label1, label2, merge_to:int):
    # merge label1 followed by label2 to merge_to
    # label2 slices must follow label1 slices, and no other labels should in between
    # if label2 is not immediately after label1, print warning
    new_labels = labels.copy()
    i = 0
    while i<len(new_labels):
        if new_labels[i] == label1:
            j = i
            while j<len(new_labels) and new_labels[j] == label1:
                j += 1  
            if j<len(new_labels) and new_labels[j] == label2:
                while j<len(new_labels) and new_labels[j] == label2:
                    j += 1
                new_labels[i:j] = merge_to
            else:
                print(f"Warning: {label1} followed by {new_labels[j]} instead of {label2} at index {i}")
            i = j
        else:
            i += 1
    
    return new_labels

def rule_based_clean_labels(labels, task):
    """ Rule based clean labels
    1. set left hand of humanoid to empty when get_human_plate_L
    2. set left hand of humanoid to plate(3) when handover_plate_L
    3. set right hand of humanoid to empty when get_cap_R
    """
    for k, v in task.items():
        if v == "get_human_plate_L":
            # set left hand of humanoid to empty
            # print('get_human_plate_L = 0', labels[:, 1:2][labels[:, 0:1] == k])
            labels[:, 1:2][labels[:, 0:1] == k] = 0
        if v == "handover_plate_L":
            # set left hand of humanoid to plate(3)
            # print('handover_plate_L = 3', labels[:, 1:2][labels[:, 0:1] == k])
            labels[:, 1:2][labels[:, 0:1] == k] = 3
        if v == "get_cap_R":
            # set right hand of humanoid to empty
            # print('get_cap_R = 0', labels[:, 3:4][labels[:, 0:1] == k])
            labels[:, 2:3][labels[:, 0:1] == k] = 0
            
    return labels              

def match_timestamps(zed_start_timestamp, react_timestamps, timestamps):
    matched_frame_id = []
    max_len = int(1e8)
    for tt in react_timestamps:
        t = tt - zed_start_timestamp
        t = np.clip(t, 0, max_len - 1)
        
        # find the frame ttt in timestamps that (ttt - zed_start_timestamp) is closest to t
        ttt = np.argmin(np.abs(timestamps - zed_start_timestamp - t))
        matched_frame_id.append(ttt)
    matched_frame_id = np.stack(matched_frame_id, axis=0)
    return matched_frame_id

def split_video(subdir_path, task, slices, split_name='splited_data'):
    zed_cap = cv2.VideoCapture(os.path.join(subdir_path, 'zed_left.mp4'))
    human_cap = cv2.VideoCapture(os.path.join(subdir_path, 'camera_human.mp4'))
    humanoid_cap = cv2.VideoCapture(os.path.join(subdir_path, 'camera_humanoid.mp4'))
    
    try:
        labels = np.load(os.path.join(subdir_path, 'react_label.pkl'), allow_pickle=True)  # (T,)
    except:
        labels = np.load(os.path.join(subdir_path, 'react_label.pkl.npy'), allow_pickle=True)  # (T,)

    labels = rule_based_clean_labels(labels, task)

    os.makedirs(os.path.join(subdir_path, split_name), exist_ok=True)
    for id, name in task.items():
        os.makedirs(os.path.join(subdir_path, split_name, name), exist_ok=True)
        for i, slice in enumerate(slices[id]):
            slice_path = os.path.join(subdir_path, split_name, name, str(i))
            os.makedirs(slice_path, exist_ok=True)
            print("slice", slice_path)
            
            zed_frames = []
            human_frames = []
            humanoid_frames = []
            zed_cap.set(cv2.CAP_PROP_POS_FRAMES, slice[0])
            human_cap.set(cv2.CAP_PROP_POS_FRAMES, slice[0])
            humanoid_cap.set(cv2.CAP_PROP_POS_FRAMES, slice[0])
            
            current_frame = slice[0]
            while zed_cap.isOpened() and human_cap.isOpened() and humanoid_cap.isOpened() and current_frame in slice:
                ret, frame = zed_cap.read()
                if ret:
                    zed_frames.append(frame)
                else:
                    break
                
                ret, frame = human_cap.read()
                if ret:
                    human_frames.append(frame)
                else:
                    break
                
                ret, frame = humanoid_cap.read()
                if ret:
                    humanoid_frames.append(frame)
                else:
                    break
                
                current_frame += 1
                
                
            hdf5_file_path = os.path.join(slice_path, 'rgb_imgs.hdf5')
            with h5py.File(hdf5_file_path, 'w') as hf:
                hf.create_dataset('zed_imgs', data=zed_frames)
                hf.create_dataset('human_imgs', data=human_frames)
                hf.create_dataset('humanoid_imgs', data=humanoid_frames)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--scenario', help='train config file path', default=0)
    args = parser.parse_args()

    SCENARIO_ID = args.scenario
    ITEM_LABEL_NAME = ["", "can", "cup", "plate", "tissue", "sponge"]
    task_list = {
        'react_data_src/react_data_1026/cheers': {0: 'idle', 1: 'pick_can_R', 2: 'cheers', 3: 'place_can_R'}, 
        'react_data_src/react_data_1224/cheers': {0: 'idle', 1: 'cheers'},
        
        'react_data_src/react_data_1026/shakehand': {0: 'idle', 1: 'handshake'},
        'react_data_src/react_data_1129/shakehand': {0: 'idle', 1: 'handshake'},
        'react_data_src/react_data_1224/shakehand': {0: 'idle', 1: 'handshake'},
        'react_data_src/react_data_1211/shakehand': {0: 'idle', 1: 'handshake'},
        
        'react_data_src/react_data_1026/thumbup': {0: 'idle', 1: 'thumbup'},
        
        'react_data_src/react_data_1030/pick_tissue': {0: 'idle', 1: 'pick_tissue_L'},
        
        'react_data_src/react_data_1030/plate': {0: 'idle', 1: 'pick_table_plate_LR', 2: 'handover_plate_L', 3: 'get_human_plate_L', 
                                   4: 'wash_plate_LR', 5: 'place_plate_L', 6: 'place_sponge_R'},
        
        'react_data_src/react_data_1030/cancel': {0: 'idle', 1: 'cancel'},
        
        'react_data_src/react_data_1129/photo': {0: 'idle', 1: 'take_photo'},
        'react_data_src/react_data_1129/spread_hand': {0: 'idle', 1: 'spread_hand'},
        'react_data_src/react_data_1129/wave': {0: 'idle', 1: 'wave'},
        'react_data_src/react_data_1211/wave': {0: 'idle', 1: 'wave'},
        
        'react_data_src/react_data_eval/dining': {0: 'idle', 1: 'cheers', 2: 'thumbup', 3: 'handshake', 4: 'pick_can_R', 5: 'place_can_R', 
                                    6: 'pick_tissue_L', 7: 'pick_table_plate_LR', 8: 'handover_plate_L', 9: 'get_human_plate_L',
                                    10: 'wash_plate_LR', 11: 'place_plate_L', 12: 'place_sponge_R'},
        'react_data_src/react_data_eval/dining_move': {0: 'idle', 1: 'cheers', 2: 'thumbup', 3: 'handshake', 4: 'pick_can_R', 5: 'place_can_R', 
                                    6: 'pick_tissue_L', 7: 'pick_table_plate_LR', 8: 'handover_plate_L', 9: 'get_human_plate_L',
                                    10: 'wash_plate_LR', 11: 'place_plate_L', 12: 'place_sponge_R'},
        'react_data_src/react_data_eval2/dining': {0: 'idle', 2: 'thumbup', 3: 'handshake', 4: 'wave', 5: 'take_photo', 6: 'spread_hand', 
                                     9: 'pick_tissue_L'},
    }
    for dir, task in task_list.items():
        for subdir in os.listdir(dir):
            subdir_path = os.path.join(dir, subdir)
            print(subdir_path)
            if not os.path.exists(os.path.join(subdir_path, 'react_label.pkl')):
                print("[WARN] No react_label.pkl found")
                continue
            
            labels = np.load(os.path.join(subdir_path, 'react_label.pkl'), allow_pickle=True)  # (T,)

            cls_slices = split_indices(labels[:, 0], task=task.keys(), prelong=0)
            motion_slices = split_indices(labels[:, -1], task=task.keys(), prelong=0)

            split_video(subdir_path, task, cls_slices, split_name='splited_data')
            split_video(subdir_path, task, motion_slices, split_name='splited_data_motion')
