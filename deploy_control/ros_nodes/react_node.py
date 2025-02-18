import time
import sys
import os
import argparse
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, ByteMultiArray, Int32

sys.path.append("../planner_motion/")

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp

from utils.plot_script import plot_3d_motion
from utils.utils import *
from utils import paramUtil
from configs import get_config
from datasets.interhuman import FEAT_DIM
import lightning as L
import scipy.ndimage.filters as filters

from os.path import join as pjoin
# from models import *
from collections import OrderedDict
from configs import get_config
from utils.preprocess import *
from utils import paramUtil
from datasets import DataModule
from models import InterGen, HumanClassifier
from train_classifier import build_model as build_cls_model
from ros_utils.react_planner import ReactPlanner, get_scenario_cls

from colorama import Fore, Back, Style
import time
from collections import deque
from queue import Queue

QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
QOS_PROFILE_ONCE = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

REACT_LABEL_COLOR = {
            "idle": Fore.WHITE,
            "cheers": Fore.BLUE,
            "thumbup": Fore.RED,
            "handshake": Fore.GREEN,
            "pick_can_R": Fore.YELLOW,
            "place_can_R": Fore.CYAN,
            "wave": Fore.BLUE,
            "take_photo": Fore.RED,
            "spread_hand": Fore.GREEN,
            "get_cap_R": Fore.MAGENTA,
            "give_cap_R": Fore.LIGHTBLACK_EX,
            "pick_stamp_R": Fore.LIGHTMAGENTA_EX,
            "stamp_R": Fore.LIGHTCYAN_EX,
            "place_stamp_R": Fore.LIGHTGREEN_EX,
            "close_lamp": Fore.LIGHTYELLOW_EX,
            "open_lamp": Fore.LIGHTRED_EX,
            "give_book_L": Fore.LIGHTMAGENTA_EX,
            "pick_tissue_L": Fore.MAGENTA,
            "pick_table_plate_LR": Fore.LIGHTBLACK_EX,
            "handover_plate_L": Fore.LIGHTMAGENTA_EX,
            "get_human_plate_L": Fore.LIGHTCYAN_EX,
            "wash_plate_LR": Fore.LIGHTGREEN_EX,
            "wash_1": Fore.LIGHTGREEN_EX,
            "wash_2": Fore.LIGHTGREEN_EX,
            "place_plate_L": Fore.LIGHTYELLOW_EX,
            "place_sponge_R": Fore.LIGHTRED_EX,
            "cancel": Fore.LIGHTWHITE_EX,
        }

REACT_FPS = 3

def get_shm_size(shape, dtype):
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    return nbytes

def load_cfg(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model_cfg = get_config(os.path.join(checkpoint_dir, "model.yaml"))
    train_cfg = get_config(os.path.join(checkpoint_dir, "train.yaml"))
    data_cfg = get_config(os.path.join(checkpoint_dir, "data.yaml"))
    model_cfg.defrost()
    model_cfg.CHECKPOINT = checkpoint_path
    return model_cfg, train_cfg, data_cfg

class SafeReciever(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self._h1_safe_sub = self.create_subscription(Int32, "unsafe_notify", 
                                 self.h1_safe_msg_callback, QOS_PROFILE)
        self._h1_safe_query = self.create_publisher(Int32, "safe_query", QOS_PROFILE)
        self._h1_safe_mode = None

    @property
    def h1_safe_mode(self):
        if self._h1_safe_mode is None:
            # default in unsafe mode, wait for safe mode
            self._h1_safe_query.publish(Int32(data=1))
            return 0
        return self._h1_safe_mode

    def h1_safe_msg_callback(self, msg):
        # conver msg to int
        self._h1_safe_mode = msg.data
        # self.get_logger().info(f"Recive H1 SAFE MODE: {self._h1_safe_mode}")
        if self._h1_safe_mode == 2:
            assert False, "H1 in emergency mode, exit!"

class FpsNode(SafeReciever):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.create_timer(1, self.count_fps)
        self.counter = {}
        self.counter_time = time.time()
    
    def count(self, key):
        self.counter[key] = self.counter.get(key, 0) + 1

    def count_fps(self):
        t = time.time() - self.counter_time
        self.counter_time = time.time()
        self.cnt = 0
        if self.h1_safe_mode != 0:
            self.get_logger().warn(f"[{self.get_name()}] [UNSAFE] FPS: {[(k, np.round(v/t, 2)) for k, v in self.counter.items()]}")
        else:
            self.get_logger().info(f"[{self.get_name()}] FPS: {[(k, np.round(v/t, 2)) for k, v in self.counter.items()]}")
        self.counter = {k: 0 for k in self.counter.keys()}

def fingers_a2s(qpos):
    qpos = np.clip(qpos, 0, 1)
    qpos = 1 - qpos
    return np.concatenate([qpos[6:12], qpos[:6]])

def fingers_s2a(state):
    right, left = state[:6], state[6:12]
    right = 1 - right
    left = 1 - left
    # if not self.to24:
    return np.concatenate([left, right])

def wrists_a2s(qs):
    qs = qs.copy()
    offset0 = 2.8
    
    qs[0] = qs[0] + offset0
    qs[1] = qs[1] - offset0

    qs[0] = np.clip(qs[0], 0.1, 4.5)
    qs[1] = np.clip(qs[1], -4.5, -0.1)

    return qs[0], qs[1]

def wrists_s2a(state):
    offset0 = 2.8
    left = state[0] - offset0
    right = state[1] + offset0 
    return np.stack([left, right])

class MaxVel:
    def __init__(self) -> None:
        self.max_vel = {"arm": 0.01, "hand": 0.1, "wrist": 0.1}
        self.max_acc = {"arm": 0.01, "hand": 0.1, "wrist": 0.1}

        self.init_vel = {"arm": 0.01, "hand": 0.1, "wrist": 0.1}
        self.init_acc = {"arm": 0.01, "hand": 0.1, "wrist": 0.1}

        self.current_vel = {"arm": 0.01, "hand": 0.1, "wrist": 0.1}
        self.current_acc = {"arm": 0.01, "hand": 0.1, "wrist": 0.1}

        self.step_scale = 1.5

    def reset(self):
        self.current_vel = self.init_vel.copy()
        self.current_acc = self.init_acc.copy()

    def step(self):
        for k in self.current_vel.keys():
            self.current_vel[k] = min(self.current_vel[k] + self.current_acc[k], self.max_vel[k])
            self.current_acc[k] = min(self.current_acc[k] * self.step_scale, self.max_acc[k])

class ReactNode(FpsNode):
 
    def __init__(self, infer_freq=10,
                 skip_skill=True,
                 no_skill=False,
                 evaluate=False,
                 state_args=[3, 3],
                 scenario="0",
                 long_horizon=False,
                 zero_hand=False):
        super().__init__("react_node")
        scenario = int(scenario)
        self.scenario = scenario
        self.long_horizon = long_horizon
        self.cls_list, self.cls_map, self.skill2cls, self.manip_name, _ = get_scenario_cls(scenario)
        
        self.init_model(infer_freq)
        self.human_pose = np.zeros((0, 36))
        self.human_hand = np.zeros((0, 12+self.num_obj*2))
        self.zero_hand = zero_hand
        self.max_history_len = self.hlen + self.plen
        self.main_loop_step = 0
        self.infer_freq = infer_freq
        self.eval = evaluate

        # - Subscribe: from Zed Camera 
        self.human_sub = self.create_subscription(
            ByteMultiArray, "human_pose/with_hand", self.human_callback, QOS_PROFILE
        )
        # - Subscribe: from robot
        self.h1_sub = self.create_subscription(
            Float32MultiArray, "all_states", self.h1_state_callback, QOS_PROFILE
        )
        # - Subscribe: from Safe Policy
        self.safe_sub = self.create_subscription(
            Float32MultiArray, "safe/safe_signal", self.safe_callback, QOS_PROFILE
        )
        # - Subscribe: from Manipulation Policy
        self.skill_sub = self.create_subscription(
            Float32MultiArray, "manip/skill_done", self.skill_callback, QOS_PROFILE
        )
        
        self.class_pred_pub = self.create_publisher(Float32MultiArray, "react/class_pred", QOS_PROFILE)
        
        # - Publish: to Manipulation Policy
        self.skill_pub = self.create_publisher(Float32MultiArray, "react/exec_skill", QOS_PROFILE_ONCE)
        # - Publish: to robot
        # self.h1_pub = self.create_publisher(Float32MultiArray, "arms_qpos", QOS_PROFILE)
        # self.h1_hand_pub = self.create_publisher(Float32MultiArray, "fingers_qpos", QOS_PROFILE)
        self.h1_pub = self.create_publisher(Float32MultiArray, "all_qpos", qos_profile=QOS_PROFILE)
        
        self.h1_state = np.zeros(10 + 12 + 2*self.num_obj)
        self.all_states_sub = self.create_subscription(Float32MultiArray, "all_states",
                                                       self.h1_state_callback, qos_profile=QOS_PROFILE)
        
        self.lazy_clear_state_history = False
        self.h1_state_history = deque(maxlen=30)
        
        # 
        cls_history_len = state_args[0]
        stable_repeat = state_args[1]
        # self.react_state = ReactState(num_obj=self.num_obj, cls_history_len=cls_history_len, stable_repeat=stable_repeat, scenario=self.scenario)
        self.react_state = ReactPlanner(num_obj=self.num_obj, cls_history_len=cls_history_len, 
                                        stable_repeat=stable_repeat, scenario=self.scenario,
                                        enable_prestart=False, long_horizon=long_horizon)
                                        # enable_prestart=True, long_horizon=long_horizon)
        self.main_loop_step = 0
        self.infer_freq = infer_freq
        self.create_timer(1./(REACT_FPS*infer_freq), self.main_loop)
        self.create_timer(1./(REACT_FPS*infer_freq), self.log_history_state)

        self.skip_skill = skip_skill
        self.no_skill = no_skill
        if skip_skill:
            self.current_skill_id = -1
            self.current_skill_st = 0
        
    def log_history_state(self):
        if self.h1_safe_mode != 0:
            return
        
        self.count("h1_state")
        if self.lazy_clear_state_history:
            self.h1_state_history.clear()
            self.lazy_clear_state_history = False
        self.h1_state_history.append(self.h1_state)
        
    def h1_state_callback(self, msg):
        h1_state = np.array(msg.data)

        if h1_state.shape[0] == 53:
            # NOTE: using command to smooth action
            real_arm_state = False
            if real_arm_state:
                h1_state[2:2+8] = h1_state[36:36+8]
            h1_state[36] = h1_state[-1]
            arm_state = np.zeros(10)
            # l_arm, l_wrists, r_arm, r_wrists
            arm_state[:4] = h1_state[2:2+4]
            arm_state[5:9] = h1_state[6:6+4]
            arm_state[4], arm_state[9] = wrists_s2a([h1_state[10], h1_state[11]])
            # left, right hand
            hand_state = fingers_s2a(h1_state[12:24])
            # one_hand_occupancy = np.zeros(self.num_obj)
            hand_occu = self.react_state.hand_occupancy_1hot
            # if self.long_horizon:
            #     self.h1_state = np.concatenate([arm_state, hand_state[:6], hand_state[6:]])
            # else:
            self.h1_state = np.concatenate([arm_state, 
                                            hand_state[:6], hand_occu[0], 
                                            hand_state[6:], hand_occu[1]])
            # self.get_logger().info(f"Hand State {arm_state.shape} {hand_state.shape} {self.h1_state.shape}")
        else:
            self.get_logger().error(f"Wrong h1 state shape {len(h1_state)}")
        # print("h1_state", self.h1_state.shape)
    
    def safe_callback(self, msg):
        pass
    
    def skill_callback(self, msg):
        if not self.react_state.exec_skill:
            self.get_logger().warn(f"No skill is executing, but received skill done signal {msg.data[0]}.")
            return
        
        return_signal = msg.data[0]

        # assert return_signal !=0, f"Skill {self.react_state.exec_skill_id} is done with signal {return_signal}."
        
        next_skill = self.react_state.done_skill(self.react_state.exec_skill_id, int(return_signal))
        self.get_logger().info(f"!!!!! Skill {msg.data[0]} is done. {self.react_state.hand_occupancy.numpy()}")
        if next_skill is not None:
            self.get_logger().info(f"However, run next Skill {next_skill}. {self.react_state.hand_occupancy.numpy()}")
            self.publish_manip_skill(next_skill)
            return
            
        # TODO: update react state
        # self.m2 = h1_init_motion[None, :1].repeat(clip_len, axis=1)
        h1_his = np.stack(self.h1_state_history, axis=0)[None, :self.hlen, ]
        assert len(h1_his.shape) == 3
        # self.get_logger().info(f"History state shape {h1_his.shape} {self.hlen}, {self.m2.shape}")
        m2 = np.pad(h1_his, ((0, 0), (self.hlen-h1_his.shape[1], 0), (0, 0)), mode='edge')
        # if not self.long_horizon:
        if True:
            # NOTE: Set humanoid hand occupancy for humanoid history state.
            hand_occupancy = self.react_state.hand_occupancy_1hot
            hand_occupancy = np.repeat(hand_occupancy[None], m2.shape[0], axis=0)
            # print("m2 shapes", m2.shape, hand_occupancy.shape)
            m2[:, :, 10 + 6: 10+6+self.num_obj] = hand_occupancy[:, 0]
            m2[:, :, 10 + 6*2+self.num_obj:10+(6+self.num_obj)*2] = hand_occupancy[:, 1]
        self.m2[:, :self.hlen] = m2
        # self.last_action = h1_init_motion[0]  # 10 + 12 = 22
        self.last_action = self.h1_state
        self.last_vel = np.zeros_like(self.last_action)
        
    def init_model(self, infer_freq):
        data_cfg = get_config("../planner_motion/configs/HH_datasets.yaml").interhuman_test
        data_cfg.defrost()
        data_cfg.D_LIST = [9] # 15
        args_mconf = "../planner_motion/configs/model_infer/classifier_ckpt.yaml"
        args_tconf = "../planner_motion/configs/infer_classifier.yaml"
        args_dtype = "_6d"
        # args_backbone = "_cross"
        args_backbone = "_seq"
    
        self.device = "cuda:0"
        class_model_cfg = get_config(args_mconf)
        react_model_cfg = get_config("../planner_motion/configs/model_train/d256x4_t300" +args_dtype+ args_backbone +".yaml")
        if self.scenario == 1:
            class_ckpt = class_model_cfg.CHECKPOINT1
        elif self.scenario == 2:
            class_ckpt = class_model_cfg.CHECKPOINT2
        elif self.scenario == 3:
            class_ckpt = class_model_cfg.CHECKPOINT3
        class_ckpt = f"../planner_motion/{class_ckpt}"
        react_ckpt = f"../planner_motion/{react_model_cfg.CHECKPOINT}"
        class_model_cfg, _, _ = load_cfg(class_ckpt)
        # self.t_dataset = HumanH1Dataset(data_cfg, react_model_cfg)
        react_model_cfg.defrost()
        react_model_cfg.CHECKPOINT = react_ckpt
        self.feat_unit_dim = FEAT_DIM[data_cfg.get("FEAT_TYPE", "6d")]


        models = self.build_model(class_model_cfg, react_model_cfg)
        self.class_model, self.react_model = models
        model_cfgs = [class_model_cfg, react_model_cfg]
        self.model_cfgs = model_cfgs

        for model, model_cfg in zip(models, model_cfgs):
            if model_cfg.CHECKPOINT:
                ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu", weights_only=False)
                for k in list(ckpt["state_dict"].keys()):
                    if "model" in k:
                        ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
                model.load_state_dict(ckpt["state_dict"], strict=False)
            model = model.to(self.device)
            model.eval()
            
            # self.t_dataset.data_statistics
            
        print("load data statistics", os.path.dirname(class_ckpt), os.path.dirname(react_ckpt))
        self.cls_normalizer = models[0].set_normalizer({"load_file": f"{os.path.dirname(class_ckpt)}/data_statistics.pkl"})
        self.normalizer = models[1].set_normalizer({"load_file": f"{os.path.dirname(react_ckpt)}/data_statistics.pkl"})
        self.cls_hd_input = class_model_cfg.HD_INPUT
        
        infer_cfg = get_config(args_tconf)
        
        self.hand_details = {}

        # self.hlen, self.plen = infer_cfg.TRAIN.HISTORY_LENGTH, infer_cfg.TRAIN.PREDICT_LENGTH
        self.hlen, self.plen = react_model_cfg.HISTORY_LENGTH, react_model_cfg.PREDICT_LENGTH
        self.cls_hlen = class_model_cfg.HISTORY_LENGTH

        self.num_obj = class_model_cfg.NUM_OBJ
        clip_len = self.hlen + self.plen
        B = 1
    
        self.apply_len = infer_freq
        h1_init_motion = self.load_h1_init(data_cfg.HAND_DIM)
        self.m1 = np.zeros((1, clip_len, 6*self.feat_unit_dim+(6+self.num_obj)*2))  # 6*6+(6+num_obj)*2
        self.m2 = h1_init_motion[None, :1].repeat(clip_len, axis=1)
        self.last_action = h1_init_motion[0]  # 10 + 12 = 22
        self.last_vel = np.zeros_like(self.last_action)

    def load_h1_init(self, hand_dim, add_occu=True):
        from utils.preprocess import load_motion_h1
        
        file_path_h1 = "../planner_motion/motion_data/motions_processed/h1/1.npy"
        file_path_h1_hand = "../planner_motion/motion_data/motions_processed/humanoid_hand/1.npy"

        h1_motion = load_motion_h1(file_path_h1)  # (T, 10)
        h1_hand, _, _ = load_hand_simple(file_path_h1_hand, add_occu, 1, self.num_obj)  # (T, 12+2*num_obj)
        if hand_dim > 0:
            h1_motion = np.concatenate([h1_motion, h1_hand], axis=-1)
        return h1_motion
        
    def build_model(self, class_cfg, react_cfg):
        class_model = build_cls_model(class_cfg)
        react_model = InterGen(react_cfg)
        return class_model, react_model
        
    def main_loop(self):
        if self.h1_safe_mode != 0:
            return
        
        self.count("Main")
        if self.skip_skill and self.current_skill_id != -1 and time.time() - self.current_skill_st > 2:
            # assert self.current_skill_id == self.react_state.exec_skill_id, f"Skill id {self.current_skill_id} != {self.react_state.exec_skill_id}"
            self.skill_callback(Float32MultiArray(data=[0]))
            self.current_skill_id = -1
            self.current_skill_st = 0
        self.main_loop_step += 1
        if self.human_pose is not None and self.human_hand is not None:
            if self.human_pose.shape[0] == self.max_history_len:
                if self.zero_hand:
                    self.human_hand = np.zeros_like(self.human_hand)
                self.react(self.human_pose, self.human_hand, self.main_loop_step%self.infer_freq==0)
            # self.react(self.human_pose, self.human_hand, True)
        else:
            # time.sleep(0.01)
            print("waiting for human pose")
            # rclpy.spin_once(self)
            
    def get_real_action(self, last_action, apply_action, max_action=0.01):
        for i in range(0, len(apply_action)):
            vec = apply_action[i] - last_action
            # vec = np.clip(vec, -max_action, max_action)
            # only clip the arm vec
            vec[:10] = np.clip(vec[:10], -max_action, max_action)
            apply_action[i] = last_action + vec
            last_action = apply_action[i]
        return last_action, apply_action

    def get_smooth_action(self, last_action, apply_action, last_vel, alpha: float = 0.8, class_id=None):
        # return last_action, apply_action, last_vel
        # log_file = open("output.txt", "a")
        # print("------------------------------------------------------------------", file=log_file)
        max_acc, max_vel = 0.1, 0.3
        max_acc, max_vel = 0.02, 0.3
        if self.react_state.sleep_count_down > 0:
            # max_acc, max_vel = 0.025, 0.05
            max_acc, max_vel = 0.05, 0.1
        else:
            if self.react_state.exec_phase == "idle":
                max_acc, max_vel = 0.005, 0.025
            else:
                max_acc, max_vel = 0.1, 0.2
        max_acc, max_vel = 0.01, 0.02
        if self.react_state.exec_phase == "idle" and class_id == 0:
            idle_right_arm = np.array([0.0, 0.0, 0.2, -0.3, 0.0])
            idle_left_arm = np.array([0.0, 0.0, -0.2, -0.3, 0.0])
            idle_plate_arm = np.array([0.0, 0.0, 0.0, -0.2, -1.0])
            idle_sponge_arm = np.array([0.0, 0.0, 0.0, -0.2, 0.0])
            idle_can_arm = np.array([0.0, 0.0, 0.0, -0.2, 0.0])
            idle_stamp_arm = np.array([0.0, 0.0, 0.0, -0.6, -1.0])
            # apply_action[i][:5] = idle_left_arm
            # apply_action[i][5:10] = idle_right_arm
            # left hand holding plate
            for i in range(0, len(apply_action)):
                if self.react_state.hand_occupancy[0] == 3:
                    apply_action[i][:5] = idle_plate_arm
                    apply_action[i][5:10] = idle_right_arm
                # right hand holding sponge
                if self.react_state.hand_occupancy[1] == 5:
                    apply_action[i][5:10] = idle_sponge_arm
                # right hand holding stamp
                if self.react_state.hand_occupancy[1] == 3:
                    if self.scenario in [1, 3]:
                        apply_action[i][5:10] = idle_stamp_arm
                    else:
                        apply_action[i][5:10] = idle_right_arm
                # right hand holding can
                if self.react_state.hand_occupancy[1] == 1:
                    apply_action[i][5:10] = idle_can_arm
        for i in range(0, len(apply_action)):
            ip_action = last_action + alpha * (apply_action[i] - last_action)
            ip_vel = ip_action - last_action
            ip_acc = ip_vel - last_vel

            # print(f"acc, {ip_acc.round(2)}", file=log_file)
            # print(f"vel, {ip_vel.round(2)}", file=log_file)
            acc = ip_acc
            acc = np.clip(ip_acc, -max_acc, max_acc)
            last_vel = np.clip(last_vel + acc, -max_vel, max_vel)
            # last_vel = last_vel + acc
            apply_action[i][:10] = last_action[:10] + last_vel[:10]

            # map 0.2~0.8 -> 0~1
            from_range = [0.2, 0.8]
            to_range = [0., 1.]
            apply_action[i][10:] = (apply_action[i][10:] - from_range[0]) / (from_range[1] - from_range[0]) * (to_range[1] - to_range[0]) + to_range[0]
            apply_action[i][10:] = np.clip(apply_action[i][10:], 0., 1.)

            # NOTE: if hand opppancy of left hand is holding plate, then set the hand action as ...
            if self.react_state.hand_occupancy[0] == 3:
                # print("try to holding plate")
                hand_act = [1]*2+[0.15]*4
                hand_act = [1]*2+[0.05, 0.15, 0.2, 0.4]
                apply_action[i][10:10+6] = hand_act
                left_wrist = -1.0
                apply_action[i][4] = left_wrist
                
            if not self.long_horizon:
                hand_occupancy = self.react_state.hand_occupancy_1hot
                apply_action[i][16:16+self.num_obj] = hand_occupancy[0]
                apply_action[i][-self.num_obj:] = hand_occupancy[1]

            last_action = apply_action[i]

        return last_action, apply_action, last_vel

    #     print("------------------------------------------------------------------\n", file=log_file)
    
    def hand_act12_to_act24(self, act):
        act = act.reshape(1, 12)

        fingers = np.zeros((act.shape[0], 24))
        # recover fingers to 24 dim

        fingers[:, [0, 1, 4, 6, 8, 10, 12, 13, 16, 18, 20, 22]] = act
        fingers[:, [2, 3, 5, 7, 9, 11, 14, 15, 17, 19, 21, 23]] = act
        fingers[:, [2]] = act[:, 1] * 1.6
        fingers[:, [3]] = act[:, 1] * 2.4
        fingers[:, [14]] = act[:, 7] * 1.6
        fingers[:, [15]] = act[:, 7] * 2.4

        # act = np.concatenate([act[:, :12], fingers], axis=1)
        return fingers
            
    def play_action(self, action):
        body = action[:, :10]
        left_hand = action[:, 10:10+6]
        right_hand = action[:, -self.num_obj-6:-self.num_obj]
        motion = np.concatenate([body, left_hand, right_hand], axis=1)
        assert motion.shape[1] == 10+12

        # TODO: recover publish motion[0, 10] to motion
        # msg = Float32MultiArray()
        # msg.data = motion.flatten().tolist()
        # self.h1_pub.publish(msg)
        msg_data = []
        for mid in range(self.apply_len):
            arm_pos = motion[mid, :10].flatten().tolist()
            hand_pos24 = self.hand_act12_to_act24(motion[mid, 10:]).flatten().tolist()
            
            h1_msg = Float32MultiArray()
            head_fix_pose = [np.pi, 5.44]
            # NOTE: this for interact with g1
            head_fix_pose = [np.pi, 5.34]
            # if self.react_state.hand_occupancy[0] == 3:  # plate
            #     lr_mask = [0.,1.] + [float(i==0) for i in self.react_state.hand_occupancy]
            if self.scenario in [0, 2]:
                lr_mask = [1.,1.] + [float(i==0 or i==3) for i in self.react_state.hand_occupancy]
            elif self.scenario in [1, 3]:
                lr_mask = [1.,1.] + [float(i==0 or i==2) for i in self.react_state.hand_occupancy]
            else:
                assert False, f"Unknown scenario {self.scenario}"
            # 3+2+10+24+4+1 = 44
            msg_data = msg_data + [1.,1.,1.] + head_fix_pose + arm_pos + hand_pos24 + lr_mask + [30.]
            
        # msg_data = msg_data[-44:]
        # msg_data[-1] = 5.

        h1_msg.data = msg_data
        self.h1_pub.publish(h1_msg)
        # print("publish arm: ", h1_msg.shape)
        
    def display_planner(self, class_id, ):

        class_name = self.cls_map
        # label = class_name[output.argmax(-1)[0]]
        label = class_name[class_id]
        
        # self.get_logger().info(f"Hand Occupancy: {self.react_state.hand_occupancy.numpy()}")
        
        # print(REACT_LABEL_COLOR[label] +"current class", label)
        # print(Fore.RESET, end="")

        frames = np.zeros((1, 420, 256, 3))
        cv2.putText(frames[0], label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.react_state.running_cls != 0:
            if self.react_state.exec_motion:
                running_task = self.cls_list[self.react_state.running_cls][0]
            elif self.react_state.exec_phase == "cancel":
                running_task = "canceling"
            else:
                running_task = self.cls_map[self.skill2cls[self.react_state.exec_skill_id]]
            # running_task = f"{self.react_state.running_cls} r: {label}"
            cv2.putText(frames[0], running_task, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.putText(frames[0], "Hand: "+str(self.react_state.hand_occupancy.numpy()), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # pos_text = [f"{v:.2f}" for v in self.hand_details["hand_pos"].flatten()]
        # cv2.putText(frames[0], " ".join(pos_text[:3]), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        # cv2.putText(frames[0], " ".join(pos_text[-3-2:-2]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        if "hand_near" in self.hand_details:
            near_text = [f"{v:.1f}" for v in self.hand_details["hand_near"].flatten()]
            cv2.putText(frames[0], " ".join(near_text[:4]), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            cv2.putText(frames[0], " ".join(near_text[4:8]), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        if "head_pos" in self.hand_details:
            head_text = [f"head: {v:.2f}" for v in self.hand_details["head_pos"].flatten()]
            cv2.putText(frames[0], " ".join(head_text), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.putText(frames[0], self.react_state.exec_phase, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        # cv2.putText(frames[0], f"stable: {stable_detect}", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.putText(frames[0], f"exec phase: {self.react_state.exec_phase}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.imshow("react cls", frames[0])
        get_key = cv2.waitKey(1)
        if get_key == ord('o'):
            self.react_state.hand_occupancy[1] = 0
        if get_key == ord('p'):
            self.react_state.hand_occupancy[0] = 0

    def remove_occupancy(self, motions):
        new_motions = torch.zeros(motions.shape[0], motions.shape[1], motions.shape[2]-self.num_obj*4).to(motions.device)
        
        new_motions[:, :, :36+6] = motions[:, :, :36+6]
        new_motions[:, :, 36+6:36+6+6] = motions[:, :, 36+6+self.num_obj:36+6+self.num_obj+6]

        new_motions[:, :, 36+6+6:(36+6+6)+10+6] = motions[:, :, 36+12+2*self.num_obj*2:(36+12+2*self.num_obj*2)+10+6]
        new_motions[0, :, (36+6+6)+10+6:(36+6+6)+10+6+6] = motions[:, :, (36+6+6)+10+6+self.num_obj:(36+6+6)+10+6+self.num_obj+6]

        return new_motions

    def react(self, history_motion, history_hand, generate_motion=True):
        """
        Motion input: 20 frames state

        """
        # get history motion
        # hm1 = extract_upper_motions_simple(history_motion.reshape(history_motion.shape[0], -1), self.t_dataset.feat_unit_dim)  # (20, 8*6)
        # self.m1[0, :] = np.concatenate([hm1, history_hand], axis=1)  # 6*8+12  (1, 20, 60)
        self.m1[0, :] = np.concatenate([history_motion, history_hand], axis=1)
        # print("motion shapes", self.m1.shape, self.m2.shape, history_hand.shape, history_motion.shape)
        motions = np.concatenate([self.m1, self.m2], axis=-1)  # (1, 20, 70)
        motions = motions[None, :]  # (1, 1, 20, 70)
        motions = torch.from_numpy(motions).to(self.device).reshape(1, self.hlen+self.plen, -1)
        mask = torch.ones((1, self.hlen+self.plen, 2), device=motions.device)
        mask[:, self.hlen:] = 0
        # motions: (B, T, 2*dim)
        motions = motions[:, :self.hlen+self.plen]
        motions, mask = motions.float(), mask.float()

        batch = OrderedDict({})
        batch["motions"] = motions.float() # (B, T, 2*dim)
        batch["mask"] = mask.float()
        batch["motion_lens"] = None
        
        # Get class model input
        class_batch = batch.copy()
        if not self.model_cfgs[0].ADD_OCCUPANCY:
            class_motions = self.remove_occupancy(motions)
        else:
            class_motions = motions
        class_batch["motions"] = class_motions[:, -self.hlen:]
        class_batch["mask"] = mask[:, -self.hlen:]
        class_batch["motion_lens"] = torch.zeros(1,1).long().to(device=self.device)
        class_batch.update(self.hand_details)

        # Get React model input
        batch["cond"] = self.normalizer.forward(motions, normal_slice=slice(0, motions.shape[-1])) # (B, T, human_dim+h1_dim)
        batch["cond_mask"] = mask[:, :, 0]
        batch["motion_lens"] = torch.zeros(1,1).long().to(device=self.device)

        # class model
        output = self.class_model.infer(class_batch)
        self.class_pred_pub.publish(Float32MultiArray(data=output.detach().cpu().flatten().tolist()))
        class_id, skill_id, change_cls, stable_detect = self.react_state.update_cls(output, no_skill=self.no_skill)
        
        class_name = self.cls_map
        # label = class_name[output.argmax(-1)[0]]
        label = class_name[class_id]

        self.display_planner(class_id)

        if (change_cls == "cancel") and self.react_state.exec_skill:
            # Cancel current skill
            self.react_state.start_skill(-1)
            self.current_skill_id = 0
            self.current_skill_st = time.time()
            # if self.
            self.lazy_clear_state_history = True
            self.skill_pub.publish(Float32MultiArray(data=[-1]))
            self.get_logger().info(f"!!!!!!!!!!!!!!!!!Cancel Skill {skill_id}.")
            
        elif self.react_state.exec_phase == "idle" or self.react_state.exec_motion:
            # Start a motion
            # react_cid = CLS2REACT[class_id]
            react_cid = self.cls_list[self.react_state.running_cls][-1]
            cid = torch.tensor([react_cid]).to(self.device)
            # generate_motion = False
            if generate_motion:
                act = self.react_motion(batch, cid)
                self.play_action(act)
        elif change_cls == "start":
            # Pre-start a skill
            assert self.react_state.exec_motion == False, f"Exec motion {self.react_state.exec_motion} is invalid."
            assert change_cls == "start", f"Change cls {change_cls} is invalid."
            self.publish_manip_skill(skill_id)
            
    def publish_manip_skill(self, skill_id):
        print("start skill id", skill_id)
        # assert False

        # assert skill_id == 10, f"Skill id {skill_id} is invalid."
        skill_id = self.react_state.start_skill(skill_id)
        if self.skip_skill:
            self.current_skill_id = skill_id
            self.current_skill_st = time.time()
        self.lazy_clear_state_history = True
        assert skill_id != -1, f"Skill id {skill_id} is invalid."
        self.skill_pub.publish(Float32MultiArray(data=[skill_id]))
        self.get_logger().info(f"Start Skill {skill_id}.")
        
        
    def react_motion(self, batch, class_id):
        # react model
        batch["class_id"] = class_id
        batch["class_mask"] = torch.ones_like(class_id)
        
        output = self.react_model.forward_test(batch)
        output = output["output"]
        # unnormalize data
        output = self.normalizer.backward(output, normal_slice=slice(0, output.shape[-1]))
        react_t = time.time()

        # print("Class time", class_t-start, "React time", react_t-class_t)
        
        apply_action = output[0, self.hlen:self.hlen+self.apply_len, -self.m2.shape[-1]:].detach().cpu().numpy()
        
        # self.last_action, apply_action = self.get_real_action(self.last_action, apply_action, max_action=.2)
        self.last_action, apply_action, self.last_vel = self.get_smooth_action(self.last_action, apply_action, self.last_vel, alpha=0.8, class_id=class_id)

        self.m2[:, :self.hlen-self.apply_len] = self.m2[:, self.apply_len:self.hlen]
        self.m2[:, self.hlen-self.apply_len:self.hlen] = apply_action
    
        return apply_action
        # return
        
    def parse_zed_msg(self, msg, res=128):
        if self.eval:
            shapes = [(1, 36+12+2*self.num_obj+1), (1,2,3), (1,2, res, res, 3), (1, 2, 5), (1, 3)]
            shapes = [(1, 36+12+2*self.num_obj+1), (1,2,3), (1, 2, 5), (1, 3)]
        else:
            shapes = [(1, 36+12), (1,2,3), (1, 2, 5), (1, 3)]
        dtype = [np.float32, np.float32, np.float32, np.float32]
        nbytes = [
            get_shm_size(s, d) for s, d in zip(shapes, dtype)
        ]
        
        b = 0
        ret = []
        # concate a list of bytes to a single bytes
        data = b"".join(msg.data)
        for i, n in enumerate(nbytes):
            d = np.frombuffer(data[b:b+n], dtype=dtype[i])
            ret.append(d.reshape(shapes[i]))
            # ret.append(np.frombuffer(msg.data[i], dtype=dtype[i]))
            b += n
        assert b == len(data), f"b {b} != len(data) {len(data)}"
        if len(ret) == 4:
            ret = ret[:2] + [None] + ret[2:]
        return ret
    
    def parse_hand_details(self, hand_pos, hand_imgs, hand_near_diou, head_pos, hand_obj=None, types=["hand_pos"]):
        num_obj = self.num_obj
        # print("hand_near raw", hand_near_diou)
        hand_near_diou = parse_diou_1hot(hand_near_diou, num_obj)
        # print("hand_near 1hot", hand_near_diou)
        if self.model_cfgs[0].get("BETTER_STATE", False):
            hand_near_diou = preprocess_hand_diou(hand_near_diou, self.model_cfgs[0].get("HAND_IOU_MEAN_POOL", False))
        
            # print("hand_near better", hand_near_diou)
        
        hand = preprocess_hand_pos(hand_pos, add_z=self.model_cfgs[0].get("HAND_POS_ADD_Z", False))
        head = preprocess_head_pos(head_pos)
        
        # (-1, 2, 128, 128, 3)->(-1, 2, 3, 128, 128)
        # BGR -> RGB
        if hand_imgs is not None:
            hand_imgs = hand_imgs[..., [2,1,0]]
            hand_imgs = np.transpose(hand_imgs, (0, 1, 4, 2, 3))
        hand_near_diou = hand_near_diou.reshape((hand.shape[0], -1))
        
        hand_objs = hand_objs.reshape((hand.shape[0], -1)) if hand_obj is not None else None
        
        ret = [hand, hand_imgs, hand_near_diou, hand_objs, head]
        ret = [r.astype(np.float32) if r is not None else None for r in ret]
        ret_dict = {}
        if "hand_pos" in types:
            ret_dict["hand_pos"] = ret[0]
        if "hand_imgs" in types:
            ret_dict["hand_imgs"] = ret[1]
        if "hand_near" in types:
            ret_dict["hand_near"] = ret[2]
            
        if "hand_obj" in types:
            ret_dict["hand_obj"] = ret[3]
        if "head_pos" in types:
            ret_dict["head_pos"] = ret[4]

        for k, v in ret_dict.items():
            ret_dict[k] = torch.from_numpy(v).to(self.device)
        return ret_dict

    def human_callback(self, msg):
        self.count("human_callback")
        # parse msg
        data, hand_pos, hand_imgs, hand_near, head_pos = self.parse_zed_msg(msg)
        self.hand_details = self.parse_hand_details(hand_pos, hand_imgs, hand_near, head_pos, types=self.cls_hd_input)
        
        if self.eval:
            self.human_pose = np.concatenate([self.human_pose, data[:, :36]], axis=0)
            human_hand = data[:, 36:36+12]
            hand_occupancy = data[:, 36+12:36+12+2*self.num_obj]  # one-hot
            human_hand = np.concatenate([human_hand[:, :6], hand_occupancy[:, :self.num_obj], human_hand[:, 6:12], hand_occupancy[:, self.num_obj:]], axis=1)
            self.human_hand = np.concatenate([self.human_hand, human_hand], axis=0)
            label = data[:, -1]
            
            self.human_pose = self.human_pose[-self.max_history_len:]
            self.human_hand = self.human_hand[-self.max_history_len:]
        else:
            self.human_pose = np.concatenate([self.human_pose, data[:, :36]], axis=0)
            human_hand = data[:, 36:]
            # if not self.long_horizon:
            if True:
                hand_occupancy = self.react_state.hand_occupancy_1hot
                hand_occupancy = np.repeat(hand_occupancy[None], data.shape[0], axis=0)
                human_hand = np.concatenate([human_hand[:, :6], hand_occupancy[:, 0], human_hand[:, 6:12], hand_occupancy[:, 1]], axis=1)
            self.human_hand = np.concatenate([self.human_hand, human_hand], axis=0)

            self.human_pose = self.human_pose[-self.max_history_len:]
            self.human_hand = self.human_hand[-self.max_history_len:]

def main(args=None):
    process_list = []

    try:
        rclpy.init()
        [p.start() for p in process_list]
        node = ReactNode(skip_skill=args.skip_skill,
                         no_skill=args.no_skill,
                         evaluate=args.eval,
                         state_args=args.state_args,
                         scenario=args.scenario,
                         long_horizon=args.long_horizon,
                         zero_hand=args.zero_hand)
        rclpy.spin(node)
        [p.join() for p in process_list]
    except KeyboardInterrupt:
        [p.terminate() for p in process_list]
        [p.join() for p in process_list]
        print(Fore.RESET)
    # finally:
    except Exception as e:
        raise e        

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--skip-skill", action="store_true")
    parse.add_argument("--no-skill", action="store_true")
    parse.add_argument("--eval", action="store_true")
    parse.add_argument("--state-args", default=[5,5], type=int, nargs="+")
    parse.add_argument("--zero-hand", action="store_true")
    parse.add_argument("--scenario", default="2", type=str)
    parse.add_argument("--long-horizon", action="store_true")
    args = parse.parse_args()
    main(args)