import numpy as np
import torch
import random

from torch.utils import data
import pickle as pkl
from utils.utils import mytqdm
from os.path import join as pjoin

from utils.utils import *
from utils.plot_script import *
from utils.preprocess import *


CLASS_MAP0 = {
        "idle": 0,
        "cheers": 1,
        "thumbup": 2,
        "handshake": 3,
        "pick_can_R": 4,
        "place_can_R": 5,
        "pick_tissue_L": 6,
        "pick_table_plate_LR": 7,
        "handover_plate_L": 8,
        "get_human_plate_L": 9,
        "wash_plate_LR": 10,
        "place_plate_L": 11,
        "place_sponge_R": 12,
        "cancel": 13,
    }
CLASS_MAP1 = {
        "idle": 0,
        "handshake": 1,
        "thumbup": 2,
        "get_cap_R": 3,
        "give_cap_R": 4,
        "pick_stamp_R": 5,
        "stamp_R": 6,
        "place_stamp_R": 7,
        "close_lamp": 8,
        "open_lamp": 9,
        "give_book_L": 10,
        "cancel": 11,
    }
CLASS_MAP2 = {
        "idle": 0,
        "cheers": 1,
        "thumbup": 2,
        "handshake": 3,
        "wave": 4,
        "take_photo": 5,
        "spread_hand": 6,
        "pick_can_R": 7,
        "place_can_R": 8,
        "pick_tissue_L": 9,
        "pick_table_plate_LR": 10,
        "handover_plate_L": 11,
        "get_human_plate_L": 12,
        "wash_plate_LR": 13,
        "place_plate_L": 14,
        "place_sponge_R": 15,
        "cancel": 16,
    }
CLASS_MAP3 = {
        "idle": 0,
        "thumbup": 1,
        "handshake": 2,
        "wave": 3,
        "take_photo": 4,
        "spread_hand": 5,
        "get_cap_R": 6,
        "give_cap_R": 7,
        "pick_stamp_R": 8,
        "stamp_R": 9,
        "place_stamp_R": 10,
        "close_lamp": 11,
        "open_lamp": 12,
        "give_book_L": 13,
        "cancel": 14,
    }
FEAT_DIM = {
            "pos": 3,
            "axis": 3, #?
            "quat": 4,
            "6d": 6
        }


def check_one_hot(data):
    """check if the data is one-hot encoded.
    """
    eps = 1e-7
    data_0 = (np.abs(data - 0) < eps).sum(axis=0)
    data_1 = (np.abs(data - 1) < eps).sum(axis=0)
    is_1hot = ((data_0 + data_1) == data.shape[0])
    return is_1hot


class HumanH1Dataset(data.Dataset):
    def __init__(self, opt, model_cfg):
        self.opt = opt
        self.model_cfg = model_cfg
        self.max_cond_length = 30
        self.min_cond_length = 30
        self.max_gt_length = 200
        self.min_gt_length = 1
        self.reverse = False
        self.data_statistics = {}
        self.feat_type=opt.get("FEAT_TYPE", "6d")
        self.feat_unit_dim = FEAT_DIM[self.feat_type]

        self.scenario = opt.SCENARIO
        if opt.SCENARIO == 0:
            self.class_map = CLASS_MAP0
        elif opt.SCENARIO == 1:
            self.class_map = CLASS_MAP1
        elif opt.SCENARIO == 2:
            self.class_map = CLASS_MAP2
        elif opt.SCENARIO == 3:
            self.class_map = CLASS_MAP3
        # print("opt.SCENARIO", opt.SCENARIO, self.class_map)
        # exit()
        
        self.delay_shift = opt.get("DELAY_SHIFT", 0)
        
        self.history_length = model_cfg.get("HISTORY_LENGTH", 1)
        self.future_length = model_cfg.get("PREDICT_LENGTH", 0)
        self.remove_history = model_cfg.get("REMOVE_HISTORY", 0)
        
        # improve hand details state
        self.obj_mask_ratio = model_cfg.get("OBJ_MASK_RATIO", 0.0)
        self.better_state = model_cfg.get("BETTER_STATE", False)
        self.hand_iou_mean_pool = model_cfg.get("HAND_IOU_MEAN_POOL", False)
        self.hand_pos_noise_std = model_cfg.get("HAND_POS_NOISE_STD", 0.)
        self.hand_pos_add_z = model_cfg.get("HAND_POS_ADD_Z", False)
        self.hand_downsample_interval = 3

        self.max_cond_length = self.history_length
        self.min_cond_length = self.history_length

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length
        
        self.max_length = self.history_length + self.future_length
        self.max_gt_length = self.history_length + self.future_length

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE
        
        self.random_swap = opt.__dict__.get("RANDOM_SWAP", True)
        
        self.load_hand_details = opt.get("LOAD_HAND_DETAILS", False)

        self.upper_body_only = opt.UPPER_BODY_ONLY
        # self.ignore_relative = opt.IGNORE_RELATIVE
        self.ignore_relative = True
        if self.upper_body_only:
            print("Dataset: upper body only")
        if self.ignore_relative:
            print("NOTE: Relative motion pos is ignored")

        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "test.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "infer":
            if os.path.exists(os.path.join(opt.DATA_ROOT, "infer.txt")):
                data_list = open(os.path.join(opt.DATA_ROOT, "infer.txt"), "r").readlines()
            else:
                data_list = [str(d)+"\n" for d in opt.D_LIST]
        elif self.opt.MODE == "train_motion":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "train_motion.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val_motion":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "val_motion.txt"), "r").readlines()
            except Exception as e:
                print(e)
                
        # data_list = data_list[:32]
        # data_list = ['3292\n'] 
        
        if not self.random_swap:
            random.shuffle(data_list)


        index = 0
        label_cnt = {}

        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT)):
            for file in mytqdm(files):
                if file.endswith(".npy") and "humanoid_hand" in root:
                    motion_name = file.split(".")[0]
                    if file.split(".")[0]+"\n" not in data_list:
                        continue
                    file_path_humanoid_hand = pjoin(root, file)
                    # file_path_humanoid = pjoin(root.replace("humanoid_hand", "humanoid"), file)
                    file_path_human = pjoin(root.replace("humanoid_hand", "human"), file)
                    file_path_human_hand = pjoin(root.replace("humanoid_hand", "human_hand"), file)
                    file_path_h1 = pjoin(root.replace("humanoid_hand", "h1"), file)
                    
                    file_path_human_hand_details = pjoin(root.replace("humanoid_hand", "human_hand_details"), file)
                    file_path_human_hand_details = file_path_human_hand_details.replace("npy", "hdf5")
                    
                    # file_path_h1_vr = pjoin(root.replace("humanoid_hand", "h1_vr"), file)
                    text_path = file_path_human.replace("human", "annots").replace("npy", "txt")

                    try:
                        # './data/annots/3.txt'
                        text = open(text_path, "r", encoding = 'latin1').readline().replace("\n", "")
                    except Exception as e:
                        print(e, text_path)
                        raise e
                    
                    # # ! no idle motions
                    # if text == "idle":
                    #     continue
                    
                    # motion: (T, D)
                    # motion1, motion1_swap = load_motion_simple(file_path_human, self.min_length, feat_types=self.feat_type)
                    # motion2, motion2_swap = load_motion_simple(file_path_humanoid, self.min_length, feat_types=self.feat_type)
                    
                    motion1 = load_motion_human(file_path_human)
                    if model_cfg.NO_HUMAN_MOTION:
                        motion1 = np.zeros_like(motion1)

                    # h1 motion: (T, 10)
                    # assert motion1.shape[1] == 24*self.feat_unit_dim, f"motion1 shape {motion1.shape} {file_path_human}"
                    assert motion1.shape[1] == 6*self.feat_unit_dim, f"motion1 shape {motion1.shape} {file_path_human}"
                    h1_motion = load_motion_h1(file_path_h1)

                    # hand motion: (T, 12 / 14)
                    hand1, obj_l, obj_r = load_hand_simple(file_path_human_hand, self.model_cfg.ADD_OCCUPANCY, 1, self.model_cfg.NUM_OBJ)
                    if model_cfg.NO_HUMAN_MOTION:
                        hand1 = np.zeros_like(hand1)
                        obj_l = 0
                        obj_r = 0
                    hand2, _, _ = load_hand_simple(file_path_humanoid_hand, self.model_cfg.ADD_OCCUPANCY, 1, self.model_cfg.NUM_OBJ)

                    # hand1 = np.zeros_like(hand1)
                    # hand2 = np.zeros_like(hand2)

                    if self.opt.MODE == "test":
                        # pad humanoid shape to human shape
                        if len(hand2) < len(hand1):
                            h1_motion = np.pad(h1_motion, ((0, len(hand1)-len(h1_motion)), (0,0)), 'constant', constant_values=0)
                            hand2 = np.pad(hand2, ((0, len(hand1)-len(hand2)), (0,0)), 'constant', constant_values=0)
                        else:
                            h1_motion = h1_motion[:len(hand1)]
                            hand2 = hand2[:len(hand1)]
                    
                    if self.load_hand_details:
                        hand_pos, hand_imgs, hand_near, _, head_pos = load_hand_details(file_path_human_hand_details, 
                                                                              self.model_cfg.NUM_OBJ,
                                                                              hand_pos_add_z=self.hand_pos_add_z)

                        if self.better_state:
                            hand_near = preprocess_hand_diou(hand_near, hand_iou_mean_pool=self.hand_iou_mean_pool)
                        
                        assert np.isnan(hand_near).sum() == 0, f"hand_near nan {hand_near}"
                        
                    
                    # # human hand obj replaced with humanoid hand obj
                    # hand1[:, 6] = hand2[:, 6]
                    # hand1[:, 13] = hand2[:, 13]

                    if motion1 is None:
                        continue

                    if self.opt.MODE == "test":
                        self.remove_history = 30-self.history_length
                        
                    remove_history = self.remove_history
                    if remove_history > 0:
                        motion1 = motion1[remove_history:]
                        h1_motion = h1_motion[remove_history:]
                        hand1 = hand1[remove_history:]
                        hand2 = hand2[remove_history:]
                        
                        if self.load_hand_details:
                            hand_pos = hand_pos[remove_history:]
                            hand_imgs = hand_imgs[remove_history:]
                            hand_near = hand_near[remove_history:]
                            head_pos = head_pos[remove_history:]
                    
                    if self.opt.FPS == 15:
                        assert False, "not implemented."
                        motion1 = motion1[::2]
                        h1_motion = h1_motion[::2]
                        hand1 = hand1[::2]
                        hand2 = hand2[::2]
                        
                    if motion1.shape[0] < self.min_length:
                        print("too short motion", motion1.shape[0], file)
                        continue
                        # exit()

                    if self.cache:
                        # self.motion_dict[index] = [motion1, motion2, h1_motion, hand1, hand2]
                        self.motion_dict[index] = [motion1, h1_motion, hand1, hand2]
                    else:
                        assert False, "not implemented."

                    label = self.class_map.get(text, -1)
                    assert label > -1, f"unknown label {text}, from {text_path}"
                    label_cnt[text] = label_cnt.get(text, 0) + 1

                    self.data_list.append({
                        # "idx": idx,
                        "name": motion_name,
                        "motion_id": index,
                        "swap":False,
                        "label":label,
                        "obj": (obj_l, obj_r),
                    })
                    if self.load_hand_details:
                        self.data_list[-1].update({
                            "hand_pos": hand_pos,
                            "hand_imgs": hand_imgs,
                            "hand_near": hand_near,
                            "head_pos": head_pos,
                        })

                    index += 1
        print("total dataset: ", len(self.data_list))
        print("label_cnt", label_cnt)
        self._count_data()
        self._statistic()
        if self.opt.MODE == "test":
            self._map_test_data()
    
    def _count_data(self):
        class_cnt = {}
        for data in self.data_list:
            label = data["label"]
            (obj_l, obj_r) = data["obj"]
            if label == 0:
                class_cnt[(label, obj_l, obj_r)] = class_cnt.get((label, obj_l, obj_r), 0) + 1
            else:
                class_cnt[label] = class_cnt.get(label, 0) + 1
        
        sorted_class_cnt = sorted(class_cnt.items(), key=lambda x: (x[0] if isinstance(x[0], int) else x[0][0]))
        for key, v in sorted_class_cnt:
            if isinstance(key, int):
                label = key
            else:
                label, obj_l, obj_r = key
            class_name = [k for k, v in self.class_map.items() if v == label][0]
            if label == 0:
                print((class_name, obj_l, obj_r), v)
            else:
                print(class_name, v)
        self.class_cnt = class_cnt


    def _statistic(self):
        if self.opt.MODE != "train" and self.opt.MODE != "train_motion":
            return
        
        # feat_names = ["motion1", "motion2", "h1", "hand1", "hand2"]
        feat_names = ["motion1", "h1", "hand1", "hand2"]
        stats = []
        for i in range(len(self.motion_dict[0])):
            stat = {}
            d = [m[i] for m in self.motion_dict.values()]  # len(d) = len(data_list)
            d = np.concatenate(d, axis=0)
            is_1hot = check_one_hot(d)
            stat["mean"] = np.mean(d, axis=0)
            stat["std"] = np.std(d, axis=0) + 1e-10
            # stat["std"] = np.maximum(stat["std"], 1e-3)
            stat["mean"] = np.where(is_1hot, 0.5, stat["mean"])
            stat["std"] = np.where(is_1hot, 1, stat["std"])
            stat["min"] = np.min(d, axis=0)
            stat["max"] = np.max(d, axis=0)
            
            stats.append(stat)
            
        for k in stat:
            stats[0][k] = np.concatenate((stats[0][k], stats[2][k]), axis=0)
            stats[1][k] = np.concatenate((stats[1][k], stats[3][k]), axis=0)
            self.data_statistics[k] = np.concatenate([stats[0][k], stats[1][k]])
            
        if self.load_hand_details:
            # hand_pos, hand_imgs, hand_near
            name_list = ["hand_pos", "hand_near", "head_pos"]
            for i, name in enumerate(name_list):
                stat = {}
                d = [dd[name] for dd in self.data_list]
                d = np.concatenate(d, axis=0)
                is_1hot = check_one_hot(d)
                self.data_statistics[f"{name}_mean"] = np.mean(d, axis=0)
                self.data_statistics[f"{name}_std"] = np.std(d, axis=0) + 1e-10
                # self.data_statistics[f"{name}_std"] = np.maximum(self.data_statistics[f"{name}_std"], 1e-3)
                self.data_statistics[f"{name}_mean"] = np.where(is_1hot, 0.5, self.data_statistics[f"{name}_mean"])
                self.data_statistics[f"{name}_std"] = np.where(is_1hot, 1, self.data_statistics[f"{name}_std"])
                self.data_statistics[f"{name}_min"] = np.min(d, axis=0)
                self.data_statistics[f"{name}_max"] = np.max(d, axis=0)
        
        if self.opt.MODE == "test":
            return
        
        # with open(os.path.join(self.opt.DATA_ROOT, "data_statistics.pkl"), "wb") as f:
        print("data_statistics", self.data_statistics.keys(), [self.data_statistics[k].shape for k in self.data_statistics.keys()])
        
        # with open(os.path.join("data_statistics_tmp.pkl"), "wb") as f:
        #     pkl.dump(self.data_statistics, f)

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        if self.opt.MODE == "val" or self.opt.MODE == "val_motion":
            return self.real_len()*10
        elif self.opt.MODE == "test":
            return len(self.test_data_map)
        else:
            return self.real_len()*100

    def process_motion(self, motion1, h1_motion, hand1, hand2):
        # motion_len = min(len(motion1), len(motion2)-self.delay_shift)
        motion_len = len(motion1)
        # gt_motion1 = motion1[:motion_len-self.delay_shift]  # (T, D)
        # gt_hand1 = hand1[:motion_len-self.delay_shift]  # (T, 12)
        # gt_hand2 = hand2[self.delay_shift:motion_len]
        # gt_h1_motion = h1_motion[self.delay_shift:motion_len]
        gt_motion1 = motion1
        gt_hand1 = hand1
        gt_hand2 = hand2
        gt_h1_motion = h1_motion

        gt_length = len(gt_motion1)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            # gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)
            # if gt_h1_motion is not None:
            gt_h1_motion = np.pad(gt_h1_motion, ((0, padding_len), (0, 0)), 'constant', constant_values=0)
            gt_hand1 = np.pad(gt_hand1, ((0, padding_len), (0, 0)), 'constant', constant_values=0)
            gt_hand2 = np.pad(gt_hand2, ((0, padding_len), (0, 0)), 'constant', constant_values=0)
        else:
            padding_len = 0

        assert len(gt_motion1) == self.max_gt_length, f"{len(gt_motion1)} != {self.max_gt_length}, {motion1.shape}"
        # assert len(gt_motion2) == self.max_gt_length, f"{len(gt_motion2)} != {self.max_gt_length}, {motion2.shape}"
        assert len(gt_h1_motion) == self.max_gt_length, f"h1 {len(gt_h1_motion)} != {self.max_gt_length}"
        assert len(gt_hand1) == self.max_gt_length, f"hand1 {len(gt_hand1)} != {self.max_gt_length}"
        assert len(gt_hand2) == self.max_gt_length, f"hand2 {len(gt_hand2)} != {self.max_gt_length}"

        info = ()

        # clip to upper body
        if self.upper_body_only:
            # whole_body motion
            # wb_motion1, wb_motion2 = gt_motion1, gt_motion2
            wb_motion1 = gt_motion1
            wb_h1_motion = gt_h1_motion
            # gt_motion1 = extract_upper_motions_simple(gt_motion1, self.feat_unit_dim)
            # gt_motion2 = extract_upper_motions_simple(gt_motion2, self.feat_unit_dim)
            # info = (wb_motion1, wb_motion2, wb_h1_motion)
            info = (wb_motion1, wb_h1_motion)

        if self.opt.HAND_DIM > 0:
            gt_motion1 = np.concatenate((gt_motion1, gt_hand1), axis=1)
            gt_h1_motion = np.concatenate((gt_h1_motion, gt_hand2), axis=1)

        # return gt_motion1, gt_motion2, gt_h1_motion, gt_length, info
        return gt_motion1, gt_h1_motion, gt_length, info, padding_len
    
    def _map_test_data(self):
        self.test_data_map = []
        for i, data in enumerate(self.data_list):
            motion_id = data["motion_id"]
            full_motion1, full_h1_motion, full_hand1, full_hand2 = self.motion_dict[motion_id]
            length = full_motion1.shape[0] - self.delay_shift
            for j in range(0, length - (self.min_gt_length+self.history_length)):
                self.test_data_map.append((i, j))

    def __getitem__(self, item):
        if self.opt.MODE != "test":
            return self._get_item(item)
        else:
            item, sub_idx = self.test_data_map[item]
            return self._get_item(item, sub_idx)
        
    def process_near_obj_state(self, hand_near):
        """ randomly mask near obj """
        hand_near = hand_near.reshape((1, -1))
        if self.obj_mask_ratio > 0 and self.opt.MODE == "train" and \
            np.random.rand() > self.obj_mask_ratio:
            hand_near[:] = 0.
            if self.better_state:
                hand_near[:, -1] = 1.
            else:
                hand_near[:, -4] = 10.
        
        return hand_near.reshape(-1)
    
    def process_hand_pos(self, hand_pos):
        """ add random gaussian noise"""
        if self.hand_pos_noise_std > 0:
            hand_pos = hand_pos + np.random.randn(*hand_pos.shape)*self.hand_pos_noise_std
        return hand_pos
    
    def downsample_hand_fps(self, hand_pos, interval=3):
        """ downsample hand pos fps """
        shift = np.random.randint(interval) # 3-> 0~2
        hand_pos = np.pad(hand_pos, ((0, shift), (0, 0)), 'edge')
        last_pos = None
        for i in range(hand_pos.shape[0]):
            ii = hand_pos.shape[0]-1 -i
            if i % interval == 0:
                last_pos = hand_pos[ii]
            hand_pos[ii] = last_pos
            
        if shift > 0:
            hand_pos = hand_pos[:-shift]
        return hand_pos

    def _get_item(self, item, sub_idx=None):
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        label = data["label"]
        
        if self.load_hand_details:
            hand_pos = data["hand_pos"]
            hand_imgs = data["hand_imgs"]
            hand_near = data["hand_near"]
            head_pos = data["head_pos"]
        else:
            hand_pos = None

        if self.cache:
            # full_motion1, full_motion2, full_h1_motion, full_hand1, full_hand2 = self.motion_dict[motion_id]
            full_motion1, full_h1_motion, full_hand1, full_hand2 = self.motion_dict[motion_id]

        
        # apply delay_shift
        if self.delay_shift > 0:
            full_motion1 = full_motion1[:-self.delay_shift]
            full_h1_motion = full_h1_motion[self.delay_shift:]
            full_hand1 = full_hand1[:-self.delay_shift]
            full_hand2 = full_hand2[self.delay_shift:]
            
        length = full_motion1.shape[0]

        # Randomly clip motion:
        # - always keep cond history is full
        # - pred length is larger than min_gt_length
        
        if self.opt.MODE in ["train_motion", "val_motion"]:
            self.max_length = 200
            self.max_gt_length = 200

        if sub_idx is not None:
            idx = sub_idx
            gt_length = self.max_length
        elif length > self.max_length:
            # assert False, f"do not use too long motion {length} > {self.max_length}"
            # idx = random.choice(list(range(length - (self.max_gt_length+self.history_length))))
            if 0 > length - (self.min_gt_length+self.history_length):
                print("length", length, self.min_gt_length+self.history_length)
            idx = random.randint(0, length - (self.min_gt_length+self.history_length))
            if self.opt.MODE in ["train_motion", "val_motion"]:
                idx = 0
            gt_length = self.max_length
        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            
        # # !!!
        # self.max_length = 200
        # self.max_gt_length = 200
        # idx = 0
        # gt_length = min(length - idx, self.max_gt_length )
            
        mask = np.zeros((self.max_length,))
        cond_mask = np.zeros((self.max_length,))
        mask[:gt_length] = 1.
        cond_mask[:self.history_length] = 1.
        mask = np.stack([mask]*2, axis=-1)
            
        motion1 = full_motion1[idx:idx + gt_length]
        # motion2 = full_motion2[idx:idx + gt_length]
        hand1 = full_hand1[idx:idx + gt_length]
        hand2 = full_hand2[idx:idx + gt_length]
        h1_motion = full_h1_motion[idx:idx + gt_length]
                
        # if self.opt.MODE != "train":
        # only downsample human hand in test mode
        hand1 = self.downsample_hand_fps(hand1, interval=self.hand_downsample_interval)

        # gt_motion1, gt_motion2, gt_h1_motion, gt_length, info = self.process_motion(motion1, motion2, h1_motion, hand1, hand2)
        gt_motion1, gt_h1_motion, gt_length, info, padding_len = self.process_motion(motion1, h1_motion, hand1, hand2)
        mask_info = (mask, cond_mask)
        
        
        if hand_pos is not None:
            if (self.scenario < 2 and label > 0 and label < 4) or (self.scenario >= 2 and label > 0 and label < 7):
                hand_pos = self.process_hand_pos(hand_pos)
            last_frame = True
            if last_frame:
                assert len(hand_pos) == len(full_motion1), f"{len(hand_pos)} != {len(full_motion1)}"
                assert idx + gt_length - 1 < len(hand_pos), f"{idx + gt_length - 1} >= {len(hand_pos)}"
                hand_pos = hand_pos[idx + gt_length - 1]
                hand_imgs = hand_imgs[idx + gt_length - 1]
                hand_near = hand_near[idx + gt_length - 1]
                head_pos = head_pos[idx + gt_length - 1]
                
            else:
                hand_pos = hand_pos[idx:idx + gt_length]
                hand_imgs = hand_imgs[idx:idx + gt_length]
                hand_near = hand_near[idx:idx + gt_length]
                head_pos = head_pos[idx:idx + gt_length]
                if padding_len > 0:
                    hand_pos = np.pad(hand_pos, ((0, padding_len), (0, 0)), 'constant', constant_values=0)
                    hand_imgs = np.pad(hand_imgs, ((0, padding_len), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
                    hand_near = np.pad(hand_near, ((0, padding_len), (0, 0)), 'constant', constant_values=0)
                    head_pos = np.pad(head_pos, ((0, padding_len), (0, 0)), 'constant', constant_values=0)

        print_check_tensor(gt_motion1, "gt_motion1")

        # 36 + 6 + 5 + 6 + 5  +  10 + 6 + 5 + 6 + 5
        # print("label", label)
        # if label == 0:
        #     for i in [0, 50, 100, 150, 180]:
        #         if gt_motion1[i, 36 + 6 + 3 - 1] == 1:
        #             print("idle with plate: ", gt_h1_motion[i, 4])
        #         else:
        #             print("idle without plate: ", gt_h1_motion[i, 4])
        
        if self.load_hand_details:
            hand_near = self.process_near_obj_state(hand_near)
            # hand_details = (hand_pos, hand_imgs, hand_near)
            ret = (name, label, gt_motion1, gt_h1_motion, gt_length, mask_info, hand_pos, hand_imgs, hand_near, head_pos)
            # print("ret", [r.shape for r in ret[1:] if isinstance(r, np.ndarray)])
            return ret
        else:
            return name, label, gt_motion1, gt_h1_motion, gt_length, mask_info