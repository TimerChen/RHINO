import rclpy
import time
import argparse
import numpy as np
from rclpy.node import Node
import rclpy.qos
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy
# import cv2
from std_msgs.msg import (
    MultiArrayLayout,
    MultiArrayDimension,
    Float32MultiArray,
    Int32
)
import h5py
import pickle as pkl
import multiprocessing as mp
from multiprocessing import shared_memory, resource_tracker
from torchvision.transforms import v2
import copy
import sys
sys.path.append("../TeleVision")
from scripts.iphone_utils import IphoneCamSHM

np.set_printoptions(precision=2, suppress=True)
# RESOLUTION = (720, 1280)
RESOLUTION = (480, 640)

CAMERA_FPS = 60
LOOP_FPS = 30
CHUNK_SIZE = 30
import sys

CONFIG_FILE_PATH_LEFT = "h1_assets/dex_retargeting/inspire_hand_left.yml"
CONFIG_FILE_PATH_RIGHT = "h1_assets/dex_retargeting/inspire_hand_right.yml"
import os

sys.path.append(os.curdir)
from teleop.constants_vuer import *
import torch
import pyzed.sl as sl
import cv2
import pickle
from collections import defaultdict, deque
from pathlib import Path

from scripts.deploy_sim import get_norm_stats, load_policy, normalize_input, merge_act, parse_id
from act.utils import preprocess_data, get_state_dim

grd_yup2grd_zup = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
hand2inspire = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = (
        sl.RESOLUTION.HD720
    )  # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = CAMERA_FPS  # Set fps at 60

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()
    return zed

class ZedCapturer:

    def __init__(self, fake=False, just_resize=False):
        if fake:
            self.zed = None
        else:
            self.zed = init_zed()
        self.runtime_parameters = sl.RuntimeParameters()
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()

        self.just_resize = just_resize
        crop_size_h=240
        crop_size_w=320

        if self.just_resize:
            img_shape = (480, 640)
            self.image_left_np = np.zeros((3, img_shape[0], img_shape[1]), dtype=np.uint8)
            self.image_right_np = np.zeros((3, img_shape[0], img_shape[1]), dtype=np.uint8)
        else:
            cropped_img_shape = (720-crop_size_h, 1280-2*crop_size_w)
            self.image_left_np = np.zeros((3, cropped_img_shape[0], cropped_img_shape[1]), dtype=np.uint8)
            self.image_right_np = np.zeros((3, cropped_img_shape[0], cropped_img_shape[1]), dtype=np.uint8)
        
        # get first image
        while True and not fake:
            _, _, success = self.capture()
            if success:
                break
        
    def capture(self):
        success = False
        if self.zed is None:
            return self.image_left_np, self.image_right_np, False
        
        crop_size_h=240
        crop_size_w=320
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            success = True
            left_image, right_image = self.image_left, self.image_right
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
            if self.just_resize:
                self.image_left_np = cv2.cvtColor(cv2.resize(left_image.get_data(), (640, 480)), cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
                self.image_right_np = cv2.cvtColor(cv2.resize(right_image.get_data(), (640, 480)), cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
                # print(self.image_left_np.shape, self.image_right_np.shape)   # (3, 480, 640)
            else:
                self.image_left_np = cv2.cvtColor(left_image.get_data()[crop_size_h:, crop_size_w:-crop_size_w], cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
                self.image_right_np = cv2.cvtColor(right_image.get_data()[crop_size_h:, crop_size_w:-crop_size_w], cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
            
        return self.image_left_np, self.image_right_np, success
    
class ZedCapturer_SHM:
    """ Zed capturer from shared memory """
    def __init__(self, shm_name="zed_img", img_shape=(2, 720, 1280, 3), just_resize=False):
        self.just_resize = just_resize
    
        self.image_left_np = np.zeros((3, 480, 640), dtype=np.uint8)
        self.image_right_np = np.zeros((3, 480, 640), dtype=np.uint8)
        self.shm_name = shm_name
        self.shm = mp.shared_memory.SharedMemory(name=shm_name)
        # NOTE: unregister shared memory, so manip_node will not delete it when exit
        resource_tracker.unregister(f"/{shm_name}", 'shared_memory')
        self.image_shm = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
        
    def capture(self):
        crop_size_h=240
        crop_size_w=320
        # rgb(720, 1280, 3)
        # crop to (720-240, 1280-320*2, 3)=(480, 640, 3)
        # rgb = np.concatenate([rgb[crop_size_h:, crop_size_w:1280-crop_size_w], rgb[crop_size_h:, 1280+crop_size_w:-crop_size_w]], axis=1)
        if self.just_resize:
            self.image_left_np = cv2.resize(self.image_shm[0], (640, 480)).transpose(2, 0, 1).copy()
            self.image_right_np = cv2.resize(self.image_shm[1], (640, 480)).transpose(2, 0, 1).copy()
        else:
            self.image_left_np = self.image_shm[0, crop_size_h:, crop_size_w:1280-crop_size_w].transpose(2, 0, 1).copy()
            self.image_right_np = self.image_shm[1, crop_size_h:, crop_size_w:1280-crop_size_w].transpose(2, 0, 1).copy()
        return self.image_left_np, self.image_right_np, True


kPi_2 = 1.57079632
RIGHT_ARM_QPOS = np.array([[0., 0.0, 1.3, kPi_2, 0]])
RIGHT_ARM_ACT = np.array([0., -0.12, 0., -0.12, 0])
LEFT_ARM_QPOS = np.array([[0., 0.0, -1.3, kPi_2, 0]])
LEFT_ARM_ACT = np.array([0., +0.12, 0., -0.12, 0])


def _raw_control_info_no_right_hand(sname, s):
    if sname in ["hand", "force"]:
        s = s[:, [1,0]]
    # NOTE: temporary set right hand to zero
    # if sname == "force" or sname == "hand":
    #     s[:, 1] = 0
    return s
    
class ActModelInference:
    def __init__(self, model_path, data_path, taskid, 
                 epid, exptid, ckpt, device="cuda:0", 
                 action_dim=24, history_stack=0,
                 state_name=None, left_right_mask=None,
                 debug_real_state=True,
                 iphone="none",
                 skill_set_size=1,
                 use_cancel_unsafe=False,) -> None:
        # current_dir = Path(__file__).parent.resolve()
        # DATA_DIR = (current_dir.parent / "data/").resolve()
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.device = device
        self.init_state = {}
        self.infer_cnt = 0
        self.iphone=iphone
        self.skill_set_size = skill_set_size
        self.use_cancel_unsafe = use_cancel_unsafe 

        self.state_name = state_name
        self.left_right_mask = left_right_mask

        self.action_dim = action_dim
        self.history_stack = history_stack
        self.qpos_dim, self.state_dim = get_state_dim(state_name, action_dim, history_stack, left_right_mask)
        self.debug_real_state = debug_real_state
        print("state dim", self.state_dim)

        self.init_model(taskid, epid, exptid, ckpt)

    def transform_state(self, img, qpos):
        transform_ops = [v2.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                        ),
                        v2.RandomPerspective(distortion_scale=0.5),
                        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                        v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),]
        transform = v2.Compose(transform_ops)
        qpos += (self.qpos_noise_std**0.5) * torch.randn_like(qpos)
        return transform(img), qpos
    
    def load_init_data(self, taskid, epid):
        RECORD_DIR = (self.data_path / "recordings/").resolve()
        episode_name = "processed_episode_" + str(epid) +".hdf5"
        task_dir, task_name = parse_id(RECORD_DIR, taskid)
        print("task_dir", task_dir)
        print("episode_name", episode_name)
        episode_path = (Path(task_dir) / "processed" / episode_name).resolve()
        print("record", episode_path)
        
        with h5py.File(str(episode_path), "r") as data:
            actions = np.array(data["qpos_action"])
            left_imgs = np.array(data["observation.image.left"])
            right_imgs = np.array(data["observation.image.right"])
            states = np.array(data["observation.state"])
        self.init_state = {"state": states, "left_imgs": left_imgs, 
                           "right_imgs": right_imgs, "actions": actions}
        
        self.replay_actions = pickle.load(open((Path(task_dir) / "processed" / "replay_actions.pkl").resolve(), "rb"))
        
        return states, left_imgs, right_imgs, actions, task_name
    
    def preprocess_stat(self, norm_stats, ):
        state_name, left_right_mask = self.state_name, self.left_right_mask
        norm_stats["qpos_std"] = preprocess_data(
            norm_stats["qpos_std"], state_name, left_right_mask, stats_mode="std"
        ).astype(np.float32)
        norm_stats["qpos_mean"] = preprocess_data(
            norm_stats["qpos_mean"], state_name, left_right_mask, stats_mode="mean"
        ).astype(np.float32)

        if self.skill_set_size > 1:
            norm_stats["qpos_mean"] = np.concatenate([
                norm_stats["qpos_mean"], 
                np.zeros((self.skill_set_size), dtype=np.float32)
            ], axis=0)
            norm_stats["qpos_std"] = np.concatenate([
                norm_stats["qpos_std"], 
                np.ones((self.skill_set_size), dtype=np.float32)
            ], axis=0)
            self.qpos_dim = self.qpos_dim+self.skill_set_size
        return norm_stats
        
    def init_model(self, taskid, epid, exptid, ckpt):    
        temporal_agg = True
        action_dim = self.action_dim

        chunk_size = CHUNK_SIZE
        
        states, left_imgs, right_imgs, actions, task_name = self.load_init_data(taskid, epid)

        # RECORD_DIR = (self.data_path / "recordings/").resolve()
        # task_dir, model_task_name = parse_id(RECORD_DIR, exptid)
        # print("find path", (self.model_path / model_task_name).resolve(), exptid)
        # exp_path, _ = parse_id((self.model_path / model_task_name).resolve(), exptid)
        exp_path = (self.model_path / exptid).resolve()

        print("find path2", exp_path,)

        norm_stat_path = Path(exp_path) / "dataset_stats.pkl"
        norm_stats = get_norm_stats(norm_stat_path)
        # import pdb; pdb.set_trace()
        norm_stats = self.preprocess_stat(norm_stats)
        
        policy_path = Path(exp_path) / f"traced_jit_{ckpt}.pt"
        timestamps = states.shape[0]
        policy = load_policy(policy_path, "cpu")
        policy.to(self.device)
        policy.eval()

        # training value
        history_stack = self.history_stack
        if history_stack > 0:
            last_action_queue = deque(maxlen=history_stack)
            for i in range(history_stack):
                if self.action_dim == 24:
                    last_action_queue.append(actions[0])
                else: # 25
                    last_action_queue.append(np.concatenate([actions[0], [0.]]))
        else:
            last_action_queue = None
        last_action_data = None
        # player = Player(dt=1 / 30)
        if temporal_agg:
            all_time_actions = np.zeros([chunk_size, chunk_size, action_dim])
            num_actions_exe = 1
        else:
            all_time_actions = np.zeros([1, chunk_size, action_dim])
            num_actions_exe = chunk_size
            
        self.history_stack = history_stack
        self.last_action_queue = last_action_queue
        self.last_action_data = last_action_data
        self.norm_stats = norm_stats
        self.policy = policy
        self.temporal_agg = temporal_agg
        self.all_time_actions = all_time_actions
        self.num_actions_exe = num_actions_exe
        self.chunk_size = chunk_size
            
    def infer(self, states, left_imgs, right_imgs, onehot_batch=None):
        if not self.debug_real_state:
            states = preprocess_data(states, self.state_name, self.left_right_mask)
        else:
            # TODO: check load init history states, the order of hand state may be wrong!!!!
            states = preprocess_data(states, self.state_name, self.left_right_mask, 
                                    process_callback=_raw_control_info_no_right_hand)
        if onehot_batch is not None:
            states = np.concatenate([states, onehot_batch], axis=1)

        history_stack = self.history_stack
        last_action_queue = self.last_action_queue
        last_action_data = self.last_action_data
        norm_stats = self.norm_stats
        policy = self.policy
        temporal_agg = self.temporal_agg
        all_time_actions = self.all_time_actions
        num_actions_exe = self.num_actions_exe
        chunk_size = self.chunk_size
        
        output = None
        act_index = 0
        t = -1
        if history_stack > 0:
            last_action_data = np.array(last_action_queue)

        # print("input states", states.shape, left_imgs.shape, right_imgs.shape)
        data = normalize_input(
            states[t], left_imgs[t], right_imgs[t], norm_stats, last_action_data, obs_dim=self.qpos_dim,
            iphone=self.iphone
        )

        # if data[0].shape[1] == 13:
            # data = (torch.cat([data[0], torch.zeros((data[0].shape[0], 4), device=data[0].device)], axis=1), data[1])

        assert states[t].dtype == np.float32, f"states[t].dtype={states[t].dtype}, last_action_data.dtype=np.float32"
        assert data[0].dtype == torch.float32, f"data[0].dtype={data[0].dtype}, data[1].dtype={data[1].dtype}"
        assert data[1].dtype == torch.float32
        
        data = [d.float() for d in data]

        # lr_imgs, states[t] = self.transform_state([left_imgs[t], right_imgs[t]], states[t])

        if temporal_agg:
            # print("model input", data[0].shape, data[1].shape, data[0].mean(), data[1].mean())
            output = (
                policy(*data)[0].detach().cpu().numpy()
            )  # (1,chuck_size,action_dim)
            # print("model output", output.shape)
            
            if False:
                # [DEBUG] save input and output
                model_inout = ([d.detach().cpu().numpy() for d in data], output)
                fname = "manip_inout.txt"
                with open(fname, "a") as f:
                    print(f"{np.round(model_inout[0][0], 2)}", file=f)
                    print(f"{np.round(model_inout[0][1], 2)}", file=f)
                    print(f"{np.round(output, 2)}", file=f)
            
            all_time_actions[:-1, :-1] = all_time_actions[1:, 1:] 
            all_time_actions[-1, :] = output
            self.infer_cnt += 1
            # self.infer_cnt = min(self.infer_cnt, 1)

            # if self.infer_cnt == 30:
            #     self.infer_cnt = 1
            # all_time_actions[[t], t : t + chunk_size] = output
            act = merge_act(all_time_actions[-self.infer_cnt:, 0])
        else:
            all_time_actions[0, :-1] = all_time_actions[0, 1:] 

            if self.infer_cnt == 0:
                print("Inference...")
                output = policy(*data)[0].detach().cpu().numpy()
                self.infer_cnt += num_actions_exe
                all_time_actions[0, :] = output
            else:
                print("Skip inference...")
            
            self.infer_cnt -= 1
            act = all_time_actions[0, 0]
            # print("output fingers", act[-12:])
            # act_index += 1
            
        # import ipdb; ipdb.set_trace()
        if history_stack > 0:
            last_action_queue.append(act)

        # print("norm_stats", norm_stats["action_std"].shape, act.shape)
        if  norm_stats["action_std"].shape[0] == 24 and act.shape[0] in [25, 26]:
            print("[WARN] For progress bar mode, quick fix norm_stats 0.5(+-0.29)")
            norm_stats["action_mean"] = np.concatenate([norm_stats["action_mean"], [0.5]], dtype=np.float32)
            norm_stats["action_std"] = np.concatenate([norm_stats["action_std"], [0.29]], dtype=np.float32)
        if self.action_dim == 26 and norm_stats["action_std"].shape[0] == 25:
        # if self.use_cancel_unsafe:
            print("[WARN] Cancel unsafe mode, quick fix norm_stats 0.5(+-0.29)")
            norm_stats["action_mean"] = np.concatenate([norm_stats["action_mean"], [0.5]], dtype=np.float32)
            norm_stats["action_std"] = np.concatenate([norm_stats["action_std"], [0.29]], dtype=np.float32)
        
        act = act * norm_stats["action_std"] + norm_stats["action_mean"]
        
        return act      
    

QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2
        )

class SkillManager():
    def __init__(self, model_path, data_path,
                 debug_real_state=False, skills_todo=None,
                 iphone="none") -> None:
        self.skills = {}
        self.model_path = model_path
        self.data_path = data_path
        self.debug_real_state = debug_real_state
        self.skills_todo = skills_todo
        self.iphone = iphone
        
        self.cfgs = {}
        if iphone != "none":
            self.load_config(f"skills_{iphone}.yaml")
        else:
            self.load_config("../TeleVision/skills.yaml")
        
    def load_config(self, config_path):
        # load yaml
        import yaml
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        default_config = config["DefaultConfig"]
        self.cfgs[-1] = default_config.copy()
        
        # load cancel pairs
        self.cancel_skill = {}
        for cancel_pair in config["SkillPairs"]:
            self.cancel_skill[cancel_pair[0]] = cancel_pair[1]
            self.cancel_skill[cancel_pair[1]] = cancel_pair[0]

        for skill_config in config["Skills"]:
            if self.skills_todo is not None and skill_config["id"] not in self.skills_todo:
                continue
            cfg = default_config.copy()
            cfg.update(skill_config)
            self.load_one_skill(cfg)
            self.cfgs[cfg["id"]] = cfg
            # try:
            #     self.load_one_skill(cfg)
            # except Exception as e:
            #     print(f"[ERROR] Load skill {cfg['id']} from {cfg['exptid']} failed", e)

    def load_one_skill(self, config):
        model_path, data_path = self.model_path, self.data_path
        skill_id = config["id"]
        exptid = config["exptid"]
        ckpt = config["ckpt"]
        state_name = config["state_name"]
        left_right_mask = config["left_right_mask"]
        action_dim = config["action_dim"]
        history_stack = config.get("history_stack", 0)
        replay_i = config["reference_replay_id"]
        delay = config["delay"]
        skill_set_size = config["skill_set_size"]
        use_cancel_unsafe = config["use_cancel_unsafe"]
        print("config", config)

        left_right_mask = left_right_mask

        replay_taskid = str(skill_id)
        act_model = ActModelInference(model_path, 
                                    data_path, 
                                    action_dim=action_dim,
                                    history_stack=history_stack,
                                    taskid=replay_taskid, 
                                    epid=replay_i, 
                                    exptid=exptid, 
                                    ckpt=ckpt, 
                                    state_name=state_name, 
                                    left_right_mask=left_right_mask,
                                    debug_real_state=self.debug_real_state,
                                    iphone=self.iphone,
                                    skill_set_size=skill_set_size,
                                    use_cancel_unsafe=use_cancel_unsafe,
                                    )
        self.skills[skill_id] = {"act_model": act_model, 
                                 "left_right_mask": left_right_mask,
                                 "action_dim": action_dim,
                                 "delay": delay,}

def fingers_s2a(state):
    right, left = state[:6], state[6:12]
    right = 1 - right
    left = 1 - left
    # if not self.to24:
    return np.concatenate([left, right])

def wrists_s2a(state):
    offset0 = 2.8
    left = state[0] - offset0
    right = state[1] + offset0 
    return np.stack([left, right])

def fingers_12to24(act):
    act = act.reshape(-1, 12)
    fingers = np.zeros((act.shape[0], 24))
    # fingers = np.random.uniform(0, 1, (act.shape[0], 24))
    # recover fingers to 24 dim

    fingers[:, [0, 1, 4, 6, 8, 10, 12, 13, 16, 18, 20, 22]] = act
    fingers[:, [2, 3, 5, 7, 9, 11, 14, 15, 17, 19, 21, 23]] = act
    fingers[:, [2]] = act[:, 13-12] * 1.6
    fingers[:, [3]] = act[:, 13-12] * 2.4
    fingers[:, [14]] = act[:, 19-12] * 1.6
    fingers[:, [15]] = act[:, 19-12] * 2.4
    return fingers

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
            return 1
        return self._h1_safe_mode

    def h1_safe_msg_callback(self, msg):
        # conver msg to int
        self._h1_safe_mode = msg.data
        self.get_logger().info(f"Recive H1 SAFE MODE: {self._h1_safe_mode}")
        if self._h1_safe_mode == 2:
            assert False, "H1 in emergency mode, exit!"

class TeleopNode(SafeReciever):
    # Subscribe Image from zed node, publish headmatrix to teleop head node, Solve IK directly
    def __init__(
        self,
        model_path,
        data_path,
        resolution=RESOLUTION,
        loop_fps=LOOP_FPS,
        state_delay=5,
        img_delay=0,
        # use_zed_shm=False,
        react_mode=False,
        iphone="none",
        scenario="dining",
        verbose=False,
        object="user",
        replay=False,
        just_resize=False,
        info="",
        enable_skill_cancel=False,
    ):
        super().__init__("Manip_node")
        self.debug_real_action = True
        self.debug_real_state = True
        self.debug_real_img = True
        # self.debug_real_state = False
        # self.debug_real_img = False
        self.replay = replay

        self.delay = (state_delay, img_delay)
        self.safe_clip_value = 0.02
        self.default_safe_act_value = 0.01
        self.safe_act_value_init = 0.001
        self.current_safe_act_value = 0

        self.last_safe_act = None

        self.resolution = resolution
        self.img_shape = (self.resolution[0], 2 * self.resolution[1], 3)

        self.loop_fps = loop_fps
        # self.reset_skill = False
        self.RESET_COUNT = 1
        self.reset_skill_count = self.RESET_COUNT
        self.enable_check_unsafe = False
        self.reset_history_h1_states = None
        self.reset_history_images = None
        self.iphone = iphone
        self.verbose = verbose
        self.scenario = scenario
        self.object = object
        
        self.react_mode = react_mode
        if self.react_mode:
            self.node_state = "waiting"
        else:
            self.node_state = "init"
        # When running in skill canceling, avoid to cancel the skill again.
        self.node_in_cancel = False
        # In Easy Cancel Mode, the skill can be canceled by the user at any time.
        self.easy_cancel = True
        self.enable_skill_cancel = enable_skill_cancel
        
        # NOTE: new manip node featrues
        self.act_prestart = True
        self.cancel_rollback = False
        # none, rollback, safe
        self.cancel_mode = "safe"

        if self.iphone != "none":
            self.iphone_cam = IphoneCamSHM(self.iphone)
        else:
            self.iphone_cam = None

        if self.react_mode:
            self.zed_capturer = ZedCapturer_SHM()
        # elif self.object.startswith("end"):
        #     self.zed_capturer = ZedCapturer_SHM(just_resize=True)
        else:
            self.zed_capturer = ZedCapturer(just_resize=just_resize)

        # for new data (after pick plate): no delay
        # self.delay = (0, 0)

        self.skill_todo = [29, 25, -1, # can: table -> table
                           26, 25, -1, # can: human -> table
                           30, 31, -1, # plate: human -> table
                           32, 33, -1, # plate: table -> human
                           30, 33, -1, # plate: human -> human
                           32, 31, -1, # plate: table -> table
                           32, 26, 33, 25, -1, # plate(table) -> can(human) -> plate(table) -> can(table)
                           ]
        self.skill_todo = [29, 25, -1]
        self.skill_todo = [30, 33, -1]
        self.skill_todo = [53, -1]
        # self.skill_todo = [50, -1]
        # self.skill_todo = [29, 25]
        # self.delay = self.skills.skills[self.current_skill]["delay"]
        # self.get_logger().info(f"Start with skill {self.current_skill}")
        # self.skill_todo = [100]
        self.skill_todo = [29, 25, 50, 32, 33, 30, 51, 31, 53]
        self.skill_todo = [54, 55, 54, 55, -1]
        self.skill_todo = [54, 55, -1]
        # self.skill_todo = [56, -1]
        # self.skill_todo = [56, 57, 58, -1]
        # self.skill_todo = [29, -1]
        # self.skill_todo = [29, 57, 58, -1]
        # self.skill_todo = [59, -1]
        # self.skill_todo = [59, 60, -1]
        # self.skill_todo = [56, 57, 58]
        # self.skill_todo = [56, 57, 58]

        # self.skill_todo = [29, 25, 50, 32, 33, 30, 51, 31, 53]

        # self.skill_todo = [61, 54]
        # pick can disturb
        # self.skill_todo = [48, 49]

        # pick and place can
        # self.skill_todo = [48, 25]
        # self.skill_todo = [33]
        # self.skill_todo = [52]
        # self.skill_todo= [33, -1]
        # get cap
        # self.skill_todo = [59, 60, -1]
        # self.skill_todo = [59, 60, 61]
        # book
        # self.skill_todo = [54, 55, -1]
        self.skill_todo = [66, -1]
        # self.skill_todo = [51, -1]
        # self.skill_todo = [200, -1]
        # self.skill_todo = [30, 31, -1]
        # self.skill_todo = [51, 52, 53, -1]
        # self.skill_todo = [52, -1]
        # self.skill_todo = [511, 52, 53, -1]

        # self.skill_todo = [70, 25, -1]
        # self.skill_todo = [71, 53, -1]
        # self.skill_todo = [70, 53, -1]
        # self.skill_todo = [45, -1]
        # self.skill_todo = [48, 25, -1]
        # self.skill_todo = [56, 57, 58, -1]
        # self.skill_todo = [46, 47, -1]
        
        object_map = {
            "user": None,
            "can": [48, 25, -1],
            "plate1": [46, 47, -1],
            "plate2": [30, 31, -1],
            "sponge": [51, 52, 53, -1],
            "tissue": [50, -1],
            "cap": [59, 60, -1],
            "cap1": [59, -1],
            "cap2": [60, -1],
            "book": [45, -1],
            "stamp": [56, -1, 57, -1, 58, -1],
            "lamp": [61, -1],
            "vio1": [63, 25, -1],
            "vio2": [56, -1, 64, -1, 58, -1],
            "vio3": [30, 65, -1],
            "end1": [66, -1],
            "end3": [67, -1],
            "end5": [68, -1],
        }
        
        if self.object in object_map:
            if self.object == "user":
                pass
            else:
                self.skill_todo = object_map[self.object]
        else:
            assert False, f"Unknown object {self.object}"
        
        self.skill_idx_in_set = 1
        
        if self.react_mode:
            if self.scenario == "dining":
                # self.skill_todo = [48, 25, 50, 32, 33, 30, 51, 31, 53]
                self.skill_todo = [48, 25, 50, 46, 47, 30, 51, 31, 53, 52]
            elif self.scenario == "office":
                # self.skill_todo = list(range(54, 61+1))
                self.skill_todo = [59, 60, 56, 57, 58, 61, 61, 45]
            else:
                assert False, f"Unknown scenario {self.scenario}"
                
        elif self.iphone != "none":
            self.skill_todo = [102]
            # self.skill_todo = [101]
        self.current_skill_todo = -1
        self.current_skill = -1

        self.skills = SkillManager(model_path, data_path, debug_real_state=self.debug_real_state, 
                                   skills_todo=self.skill_todo,
                                   iphone=self.iphone,)
        
        if self.object == "user" or True:
            self.scoreboard = None

        self.h1_state = None
        self.history_h1_states = []
        self.history_images = []
        self.history_act = []

        if not self.react_mode:
            self.switch_skill(force_reset=(len(self.skill_todo) > 1))
        else:
            # add an fake state for debug
            self.h1_state = np.zeros((53,), dtype=np.float32)

        self.create_timer(1./self.loop_fps, self.log_history_states)

        self.cnt = 0
        self.global_cnt = 0
        self.create_timer(1, self.get_fps)

        self.qos_profile = QOS_PROFILE

        self.teleop_pub = self.create_publisher(Float32MultiArray, "all_qpos", qos_profile=self.qos_profile)
        self.teleop_msg = Float32MultiArray()
        """ publish shape: (44,) = 3(head/arm/fingers mask) + 2(head yp) + 10(arm,wrist,arm,wrist) + 24(fingers) + 1(fps)"""
        """ (old) state shape: (36,) = 2(head) + 10(arm,arm,wrist,wrist) + 12(left hand) + 12(right hand) """
        teleop_msg_pub = MultiArrayDimension(label="dim", size=36)
        self.teleop_msg.layout = MultiArrayLayout(
            dim=[teleop_msg_pub], data_offset=0
        )

        self.all_states_sub = self.create_subscription(
            Float32MultiArray, "all_states", self.all_states_callback, qos_profile=self.qos_profile
        )
        
        # Subscripution: From React model
        self.skill_start_sub = self.create_subscription(
            Float32MultiArray, "react/exec_skill", self.skill_signal_callback, qos_profile=self.qos_profile
        )
        
        # Publisher: skill done
        self.skill_done_pub = self.create_publisher(Float32MultiArray, "manip/skill_done", qos_profile=self.qos_profile)
        
        # Publisher: cancel unsafe
        self.cancel_unsafe_pub = self.create_publisher(Float32MultiArray, "manip/cancel_unsafe", qos_profile=self.qos_profile)
        
        # self.h1_state = self.act_model.init_state["state"][0]
        
        self.is_warm_up = False

        self.last_skill_time = time.time()

        # self.sleep = self.react_mode
        
    def warm_up(self):
        if self.is_warm_up:
            return
        self.is_warm_up = True
        if self.react_mode:
            self.skill_done(signal="done")

    def switch_to_cancel(self):
        # seek for cancel skill.
        cancel_skill = self.skills.cancel_skill.get(self.current_skill, -1)
        print("!!current skill", self.current_skill)
        print("!!cancel skill", cancel_skill)
        print("easy cancel", self.easy_cancel)
        print("enable skill cancel", self.enable_skill_cancel)
        print("cancel mode", self.cancel_mode)
        print("node in cancel", self.node_in_cancel)
        if not self.enable_skill_cancel and not self.easy_cancel:
            # goto condition1, ignore easy cancel
            pass
            # TODO: or disable cancel
            return
    
        if self.node_in_cancel:
            # avoid cancel the cancel skill
            return
            
        if cancel_skill == -1 or self.easy_cancel or (not self.enable_skill_cancel):
            # if no cancel skill, directly cancel
            self.skill_todo = [-1]
            # self.current_skill_todo = -1
            # self.skill_done()
            self.set_state("canceling")
            if self.cancel_mode == "none":
                # directly cancel
                self.switch_skill(force_reset=True, signal="cancel")
            elif self.cancel_mode == "safe":
                self.cancel_todo = [self.react_start_act]
            else:
                assert self.cancel_mode == "rollback", f"Unknown cancel mode {self.cancel_mode}"
                self.cancel_todo = self.history_act[::-2] + [self.react_start_act]

            print("\033[91m" + f"Skill Cancel", "\033[0m")
        elif cancel_skill != -1 and (not self.easy_cancel) and self.enable_skill_cancel:
            # switch to cancel_skill
            self.skill_todo = [cancel_skill, -1]
            self.current_skill_todo = -1
            self.sleep = False
            self.set_state("running")
            self.node_in_cancel = True
            # Log start action for cancel
            self.switch_skill(force_reset=False)
            # self.react_start_act = self.action_from_state()
            self.get_logger().info(f"Cancel! switch to skill {self.skill_todo}")
        
    def skill_signal_callback(self, msg):
        if not self.react_mode:
            return

        self.get_logger().info(f"Skill signal received, switch skill {msg.data}")
        if msg.data[0] > 0:
            trans_id = {1: 29, 2: 25} # pick & place can
            # pick & place can, 3: pick plate, 4: handover plate, 5: pick_human plate, 6: pick sponge, 7: place plate, 8: place sponge
            # 1:pick_can_R, 2:place_can_R, 3:pick_tissue_L, 
            # 4:pick_table_plate_LR, 5: handover_plate_L, 
            # 6: get_human_plate_L, 7: wash_plate_LR, 8: place_plate_L, 9: place_sponge_R
            
            if self.scenario == "dining":
                trans_id = {1: 48, 2: 25, 3: 50,
                        4: 46, 5: 47, 6:30, 7:51, 8:31, 9:53, 10:52} 
            elif self.scenario == "office":
                trans_id = {1: 59, 2: 60, 3: 56, 4: 57, 5: 58, 6: 61, 7: 61, 8: 45}
            else:
                assert False, f"Unknown scenario {self.scenario}"
            
            # print with red color
            print("\033[91m" + f"Skill signal received {msg.data[0]}, switch", "\033[0m")
            
            self.skill_todo = [trans_id[msg.data[0]], -1]
            self.current_skill_todo = -1
            self.sleep = False
            self.set_state("init")
            self.easy_cancel = True
            self.node_in_cancel = False
            # Log start action for cancel
            self.switch_skill(force_reset=True)
            self.react_start_act = self.action_from_state()
            self.get_logger().info(f"Skill signal received, switch skill {self.skill_todo}")
        else:
            self.switch_to_cancel()
            self.get_logger().info(f"Receive signal {msg.data[0]}")

            
    def skill_done(self, signal):
        print("Skill done, publish signal", signal)
        print("node in cancel", self.node_in_cancel)
        if self.node_in_cancel:
            signal = "cancel"
        signal_map = {
            "done": 0.,
            "timeout": 1.,
            "cancel": 2.,
        }
        self.set_state("waiting")
        msg = Float32MultiArray(data=[signal_map[signal]])
        self.skill_done_pub.publish(msg)
        self.sleep = True
        self.last_safe_act = None
        self.reset_skill = False
        self.reset_skill_count = 0
        self.enable_check_unsafe = False
        self.react_start_act = None
        self.history_act = []
        self.teleop_msg.data = []

    @property
    def act_model(self) -> ActModelInference:
        return self.skills.skills[self.current_skill]["act_model"]
    
    @property
    def skill_cfg(self):
        return self.skills.cfgs[self.current_skill]

    def get_fps(self):
        self.get_logger().info(f"FPS: {self.cnt}, Node state {self.node_state}")
        self.cnt = 0

    def switch_skill(self, force_reset=False, signal="done"):
        self.get_logger().info(f"==== Enter switch skill, current is {self.skill_todo[self.current_skill_todo]} ====")

        self.last_skill_time = time.time()
        self.global_cnt = 0
        self.current_skill_todo = (self.current_skill_todo + 1) % len(self.skill_todo)
        self.current_skill = self.skill_todo[self.current_skill_todo]
        if self.current_skill == -1:
            if self.react_mode:
                self.skill_done(signal)
                return
            else:
                force_reset = True
                self.current_skill_todo = (self.current_skill_todo + 1) % len(self.skill_todo)
                self.current_skill = self.skill_todo[self.current_skill_todo]
                
        # if self.replay != -1:
        self.chosen_action_idx = np.random.randint(0, len(self.act_model.replay_actions))
        
        self.skills.skills[self.current_skill]["act_model"].infer_cnt = 0
        self.delay = self.skills.skills[self.current_skill]["delay"]
        self.get_logger().info(f"==== Switch to skill {self.current_skill} ====")
        if self.current_skill_todo in self.replay:
            self.debug_real_action = False
        else:
            self.debug_real_action = True

        self.current_safe_act_value = self.safe_act_value_init
        self.enable_check_unsafe = True
        if force_reset:
            self.reset_to_init()
    
    def reset_to_init(self):
        shift = self.skill_cfg["init_shift"]
        state_delay, img_delay = self.delay
        chunk_size1 = self.loop_fps + state_delay
        chunk_size2 = self.loop_fps + img_delay
        self.reset_history_h1_states = self.act_model.init_state["state"][shift:shift+1].repeat(chunk_size1, axis=0)
        left_imgs = self.act_model.init_state["left_imgs"][shift:shift+1]
        right_imgs = self.act_model.init_state["right_imgs"][shift:shift+1]
        self.reset_history_images = [(left_imgs[0], right_imgs[0]) for i in range(chunk_size2)]
        self.reset_skill = True
        self.reset_skill_count = self.RESET_COUNT
        self.set_state('init')

    def log_history_states(self):
        if self.h1_safe_mode != 0:
            return
        
        self.warm_up()
        if self.h1_state is None:
            return
        if self.scoreboard is not None:
            # sb_start = time.time()
            self.scoreboard.handle_keyboard_events()
            # print("scoreboard time", time.time() - sb_start)

        state_delay, img_delay = self.delay

        if len(self.history_h1_states) > 0 and self.history_h1_states[-1].shape[0] != self.h1_state.shape[0]:
            for i, s in enumerate(self.history_h1_states):
                if s.shape[0] < self.h1_state.shape[0]:
                    self.history_h1_states[i] = np.concatenate([s, self.h1_state[self.h1_state.shape[0]:]], axis=0)
                else:
                    self.history_h1_states[i] = s[:self.h1_state.shape[0]]
        self.history_h1_states.append(self.h1_state.copy())
        if self.iphone != 'none':
            # img, lowres_depth = ...
            # depth = ...
            img, depth = self.iphone_cam.get_image()
            self.history_images.append((img, depth))
        else:
            left_img, right_img, success = self.zed_capturer.capture()
            self.history_images.append((left_img, right_img))

        if self.debug_real_action:
            if len(self.history_h1_states) > self.loop_fps + state_delay:
                self.history_h1_states = self.history_h1_states[-(self.loop_fps+state_delay):]
                self.history_images = self.history_images[-(self.loop_fps+img_delay):]
            
            if len(self.history_h1_states) == self.loop_fps + state_delay:
                self.main_loop()
        else:
            self.main_loop()

    def apply_lr_mask(self, act):
        # return act
        if len(self.teleop_msg.data) == 0:
            return act
        if self.act_model.left_right_mask[1] == 0:
            # mask right arm and hand
            act[0, 7:7+5] = self.teleop_msg.data[7:7+5]
            act[0, 24:36] = self.teleop_msg.data[24:36]
            
        elif self.act_model.left_right_mask[0] == 0:
            # mask left 
            # arm and wrist
            act[0, 2:2+5] = self.teleop_msg.data[2:2+5]
            # fingers
            act[0, 12:24] = self.teleop_msg.data[12:24]
        return act

    def act24_to_act36(self, act):
        # print("act shape", act.shape)
        act = act.reshape(1, 24)

        # cover right
        # NOTE: directly control right arm

        # if self.act_model.left_right_mask[1] == 0:
        #     # mask right arm and hand
        #     act[:, 7:12] = np.tile(RIGHT_ARM_ACT, (act.shape[0], 1))
        #     act[:, 18:24] = np.zeros((act.shape[0], 6))
        # elif self.act_model.left_right_mask[0] == 0:
        #     # mask left arm and hand
        #     act[:, 2:2+5] = np.tile(LEFT_ARM_ACT, (act.shape[0], 1))
        #     act[:, 12:12+6] = np.zeros((act.shape[0], 6))

        fingers = fingers_12to24(act[:, 12:12+12])

        act = np.concatenate([act[:, :12], fingers], axis=1)
        return act
    
    def check_unsafe_act(self):
        # Only check unsafe in react mode
        # if (not self.enable_check_unsafe) or (not self.react_mode):
        if (not self.enable_check_unsafe) or (not self.debug_real_action):
            return False
        
        # if not self.reset_skill:
        if not self.node_state in ["init", "safe_init", "canceling", "safe_canceling"]:
            assert self.last_safe_act is None, f"now state {self.node_state},"
            assert False
            return False
        
        if self.h1_state is None or len(self.teleop_msg.data) == 0:
            return False
        
        # get the goal of safe act
        if self.last_safe_act is None:
            arm_state = (self.h1_state.data[2:2+4], self.h1_state.data[6:6+4])
            arm_state = [np.array(a) for a in arm_state]
        else:
            arm_state = self.last_safe_act

        data = self.teleop_msg.data[3:3+36]
        arm_act = (data[2:2+4], data[7:7+4])
        arm_act = [np.array(a) for a in arm_act]
       
        cvalue = self.safe_clip_value
        lr_mask = self.act_model.left_right_mask  # (2, )

        if (lr_mask[0] and np.abs(arm_act[0] - arm_state[0]).max() > cvalue) or \
            (lr_mask[1] and np.abs(arm_act[1] - arm_state[1]).max() > cvalue):
            # with open("unsafe_act.txt", "a") as f:
            #     print("[WARN] Unsafe act with reset count", self.reset_skill_count, "at skill", self.current_skill, file=f)
            # print("[WARN] Unsafe act", arm_act, arm_state)
            if self.node_state == "init":
                self.set_state("safe_init")
            elif self.node_state == "canceling":
                self.set_state("safe_canceling")

            self.pub_safe_act(arm_act, arm_state)
            return True

        # unsafe check done, reset last_safe_act
        self.last_safe_act = None
        self.enable_check_unsafe = False
        if self.node_state == "safe_init":
            self.set_state("init")
        elif self.node_state == "safe_canceling":
            self.set_state("canceling")
        else:
            assert self.node_state in ["canceling", "init"], f"Unknown node state {self.node_state}"

        return False
        
    @property
    def safe_act_value(self):
        # if self.current_skill in [33]:
        if self.current_skill in [33]:
            return 0.003
        return self.default_safe_act_value
    
    @property
    def clip_act_value(self):
        return self.current_safe_act_value
    
    def update_safe_act_value(self, step=1.1):
        self.current_safe_act_value = np.minimum(self.safe_act_value, self.current_safe_act_value * step)
        
    def action_from_state(self, lr_mask=None):
        arm_state = [self.h1_state[2:2+4], self.h1_state[6:6+4]]
        # arm_state = [np.array(a) for a in arm_state]
        # finger_state = (self.h1_state.data[12:12+12], self.h1_state.data[24:24+12])
        # wrist_state = (self.h1_state.data[10:11], self.h1_state.data[11:12])
        wrist_state = wrists_s2a(self.h1_state[10:12])
        # NOTE: finger: [right, left -> left, right]
        # finger_state = (self.h1_state.data[24:24+12], self.h1_state.data[12:12+12])

        finger_state = fingers_s2a(self.h1_state[12:24])
        finger_state = fingers_12to24(finger_state)[0]
        if lr_mask is None:
            lr_mask = self.act_model.left_right_mask * 2  # (2, )
        # msg.data = [1.]*3 + act.tolist() + [float(i) for i in lr_mask] + [30.]
        # print("arm_state", arm_state, wrist_state, finger_state, lr_mask)
        # print("all types", type(arm_state), type(wrist_state), type(finger_state), type(lr_mask))
        # nparray: arm_state, finger_state 
        act = [0., 1., 1.] + [0., 0.] + arm_state[0].tolist() + [wrist_state[0]] + arm_state[1].tolist() + [wrist_state[1]] + \
            finger_state.tolist() + [float(i) for i in lr_mask] + [30.]
        assert len(act) == 44, f"act length {len(act)}, all states {len(arm_state)}, {len(wrist_state)}, {len(finger_state)}, {len(lr_mask)}"
        
        return act

    def pub_safe_act(self, arm_act, arm_state):
        if self.last_safe_act is None:
            self.last_safe_act = [np.zeros(4), np.zeros(4)]

        cvalue = self.clip_act_value
        self.update_safe_act_value()

        data = self.teleop_msg.data[3:3+36]
        act = np.array(data)
        # act[2:2+4] = np.clip(act[2:2+4], -0.1, 0.1)
        self.last_safe_act[0] = act[2:2+4] = np.clip(arm_act[0] - arm_state[0], -cvalue, cvalue) + arm_state[0]
        self.last_safe_act[1] = act[7:7+4] = np.clip(arm_act[1] - arm_state[1], -cvalue, cvalue) + arm_state[1]
        
        lr_mask = self.act_model.left_right_mask * 2  # (2, )
        msg = Float32MultiArray()
        msg.data = [1.]*3 + act.tolist() + [float(i) for i in lr_mask] + [30.]

        # NOTE: safe init do not move fingers
        msg.data[-3] = 0.
        msg.data[-2] = 0.
        
        if self.verbose:
            print("[WARN] Unsafe act, publish safe act", act[2:2+4], act[7:7+4], self.current_safe_act_value)
        # self.teleop_pub.publish(msg)  
        self.pub_msg(msg)

    def pub_msg(self, msg):
        # (head mask, arm mask, hand mask, act, lr_mask(l_arm, r_arm, l_hand, r_hand), speed)
        if self.react_mode and not (self.node_state in ['canceling', 'waiting']):
            self.history_act.append(copy.copy(msg.data))
            self.history_act = self.history_act[-1000:]
        for i in range(10):
            msg.data[3+2+i] = msg.data[3+2+i] + self.skill_joint_offset_in_action[i]
        # msg.data[3+2:3+2+10] = msg.data[3+2:3+2+10] + self.skill_joint_offset[:]
        assert len(msg.data) == 44, f"act length {len(msg.data)}"
        self.teleop_pub.publish(msg)

    @property
    def progress_threshold(self):
        sid = self.current_skill
        return self.skill_cfg["progress_thresh"]
        if sid in [52]:
            return 1.95
        return 0.85
    
    @property
    def skill_timeout(self):
        sid = self.current_skill
        return self.skill_cfg["timeout"]
        # if sid in [61, 57, 52]: # press lamp/stamp, brush plate
        if sid in [52]:
            return 10
        
        return 15
    
    @property
    def skill_progress_frame_replay(self):
        return self.skill_cfg["progress_frame_replay"]
    
    @property
    def warmup_time(self):
        return self.skill_cfg["warmup_time"]
    
    @property
    def skill_embedding(self):
        sid = self.current_skill
        onehot = None
        
        if sid in [70]:
            assert self.skill_idx_in_set < 4
            onehot = np.zeros(4, dtype=np.float32)
            onehot[self.skill_idx_in_set] = 1
  
        if sid in [71]:
            assert self.skill_idx_in_set < 2
            onehot = np.zeros(2, dtype=np.float32)
            onehot[self.skill_idx_in_set] = 1

        return onehot
    
    @property
    def skill_joint_offset_in_action(self):
        return self.skill_cfg["global_offset"] + self.skill_cfg["offset"]
    
    @property
    def skill_joint_offset_in_state(self):
        offset = np.array(self.skill_joint_offset_in_action)
        return offset[[0,1,2,3,5,6,7,8,4,9]]
    
    def set_state(self, s):
        # if s in ["running",]:
            # self.skill_todo = [-1]
            # self.set_state("canceling")
            # self.cancel_todo = self.history_act[::-1] + [self.react_start_act]
            # return
        if self.node_state != s:
            self.get_logger().info(f"Node State: {self.node_state} -> {s}")
        self.node_state = s

    def main_loop(
        self,
    ):
        self.cnt += 1
        self.global_cnt += 1
        
        # if self.sleep or self.node_state == "waiting":
        if self.react_mode and self.node_state == "waiting":
            return
        
        if self.node_state in ["canceling", "safe_canceling"]:
            print("cancel_todo", len(self.cancel_todo))
            if len(self.cancel_todo) == 0:
                self.switch_skill(force_reset=True, signal="cancel")
                return
            if self.node_state != "safe_canceling":
                act = self.cancel_todo.pop(0)
                self.teleop_msg.data = act
            if self.check_unsafe_act():
                self.set_state("safe_canceling")
            else:
                self.pub_msg(self.teleop_msg)
            return
            # act = self.apply_lr_mask(act)
        
        # NOTE: also prepare start actions for ACT
        if not self.act_prestart:
            if self.last_safe_act is not None and self.check_unsafe_act():
                self.get_logger().info(f"Node State: {self.node_state} -> running")
                self.set_state("safe_init")
                return

        echo_model_info = False
        if self.current_skill_todo in self.replay:
            start_i = self.global_cnt % (self.act_model.replay_actions[self.chosen_action_idx].shape[0])
            # start_i = self.global_cnt % (self.act_model.replay_actions[self.chosen_action_idx].shape[0])
        else:
            start_i = self.global_cnt % (self.act_model.init_state["state"].shape[0]-CHUNK_SIZE)
        h1_state, left_imgs, right_imgs, onehot_task_id = self.get_model_input(start_i, echo_model_info)
        
        act, progress, cancel_unsafe = self.get_model_output(h1_state, left_imgs, right_imgs, onehot_task_id, start_i, echo_model_info)

        # apply act into teleop_msg
        # act = self.apply_lr_mask(act)
        if self.node_state != "safe_init" or (not self.react_mode):
            # in safe_init mode, the action msg is not updated
            lr_mask = self.act_model.left_right_mask*2
            self.teleop_msg.data = [1.]*3 + act.reshape(-1).tolist() + [float(i) for i in lr_mask]+ [30.]

        # if self.reset_skill:
        if self.node_state in ["init", "safe_init"]:
            if self.check_unsafe_act():
                return
            self.reset_skill_count -= 1
            if self.reset_skill_count <= 0:
                # self.reset_skill = False
                self.set_state("running")
        if self.node_state == "running":
            self.easy_cancel = False
            # if cancel_unsafe > 0.9:
            #     self.easy_cancel = False
        
        self.pub_msg(self.teleop_msg)
        # self.teleop_pub.publish(self.teleop_msg)
        if self.verbose:
            print("PUBLISH ACT: ", np.round(self.teleop_msg.data, 2))
        
        current_skill_span = time.time() - self.last_skill_time
        # print(f"[main_loop] progress {progress}/{self.progress_threshold} time {current_skill_span:.4f}/({self.warmup_time},{self.skill_timeout})")
        if (current_skill_span > self.warmup_time) and \
            ((progress > self.progress_threshold) or current_skill_span > self.skill_timeout):
            signal = "done" if progress > self.progress_threshold else "timeout"
            self.get_logger().info(f"Skill done, progress {progress}, signal {signal}")

            # skill end.
            if self.scoreboard is not None:
                self.scoreboard.count_time_end(self.skill_todo[self.current_skill_todo])

            self.switch_skill(signal=signal)

            # skill start
            if self.scoreboard is not None: 
                self.scoreboard.count_time_start(self.skill_todo[self.current_skill_todo])
                    

    def rule_based_act(self, act):
        # if self.current_skill in []:
        # if self.current_skill in [51, 52, 53]:
        # if self.current_skill in [30, 31, 32, 33]:
        # if self.current_skill in [ 33]:
        # if self.current_skill in [30] or (self.current_skill in [31] and time.time() - self.last_skill_time < 2):
        if self.current_skill in [30]:
            # set four fingers to 0.15, do not totally open it
            # act[12+2:12+6] = [0.05, 0.15, 0.25, 0.3]
            act[12+5:12+6] = np.clip(act[12+5:12+6], 0.4, 1)
            act[12+4:12+5] = np.clip(act[12+4:12+5], 0.2, 1)
            act[12+3:12+4] = np.clip(act[12+3:12+4], 0.15, 1)
            act[12+2:12+5] = np.clip(act[12+2:12+5], 0.05, 1)

            act[12+2:12+6] = [0.05, 0.15, 0.2, 0.4]
            
        return act

    def get_model_output(self, h1_state, left_imgs, right_imgs, onehot_task_id, start_i, echo_model_info=False):
        chunk_size = CHUNK_SIZE

        if onehot_task_id is not None:
            onehot_batch = np.tile(onehot_task_id, (h1_state.shape[0], 1))
        else:
            onehot_batch = None
        # import pdb; pdb.set_trace()
        
        if self.debug_real_action:
            if (self.skill_cfg["init_act"] and self.node_state in ["init"]):
                act = self.act_model.init_state["actions"][self.skill_cfg["init_shift"]]
            else:
                act = self.act_model.infer(h1_state, left_imgs, right_imgs, onehot_batch)
        else:
            if self.current_skill_todo in self.replay:
                if self.verbose:
                    print("Now replay data.")
                act = self.act_model.replay_actions[self.chosen_action_idx][start_i]
                n = self.act_model.replay_actions[self.chosen_action_idx].shape[0]#-chunk_size
                # progress = 0 if start_i <= n-self.skill_progress_frame_replay else 1
                progress = 0 if start_i < n-5 else 1
                # progress = start_i / n
                # print("start_i", start_i, "n", n, "progress", progress)
                # import pdb; pdb.set_trace()
                
            else:
                act = self.act_model.init_state["actions"][start_i]
                n = self.act_model.init_state["state"].shape[0]-chunk_size
                progress = 0 if start_i < n-1 else 1
        
        if echo_model_info:
            print("act: ", act)
            print("init act: ", self.act_model.init_state["actions"][start_i])
            print("\n")

        if act.shape[0] == 25:
            progress = act[-1]
            cancel_unsafe = 0  # 0:skill can be cancelled by human, 1:skill cannot be cancelled by human (cancel is unsafe)
        elif act.shape[0] == 26:
            progress = act[-2]
            cancel_unsafe = act[-1]
            
        # elif not self.debug_real_action:
        #     progress = start_i / (self.act_model.init_state["state"].shape[0]-chunk_size)
        #     cancel_unsafe = 1
        elif self.debug_real_action:
            progress = 0
            cancel_unsafe = 0
        else:
            progress = progress
            cancel_unsafe = 0
        
        if self.verbose:
            if self.current_skill_todo in self.replay:
                self.get_logger().info(f"progress {progress:.4f}/{self.progress_threshold}, {start_i}/{self.act_model.replay_actions[self.chosen_action_idx].shape[0]} {(time.time() - self.last_skill_time):.4f}/{self.warmup_time}, {self.skill_timeout}")
            else:
                self.get_logger().info(f"progress {progress:.4f}/{self.progress_threshold}, {start_i}, {self.act_model.init_state['state'].shape[0]-chunk_size}")

        act = self.rule_based_act(act)
        act = self.act24_to_act36(act[:24])

        # publish model output
        if self.verbose:
            print(
                "task", self.current_skill, 
                "progress", f"progress {progress:.4f}/{self.progress_threshold}", 
                "time", f"{(time.time() - self.last_skill_time):.4f}/({self.warmup_time}, {self.skill_timeout})"
            )
        return act, progress, cancel_unsafe


    def get_model_input(self, start_i, echo_model_info=False):
        chunk_size = CHUNK_SIZE
        state_delay, img_delay = self.delay

        # if self.reset_skill:
        if self.node_state in ["init", "safe_init"]:
            his_state = self.reset_history_h1_states
            his_imgs = self.reset_history_images
        else:
            if len(self.history_h1_states) > 0 and self.debug_real_state:
                his_state = self.history_h1_states
            else:
                his_state = self.act_model.init_state["state"][start_i:start_i+chunk_size+state_delay]

            if len(self.history_images) > 0 and self.debug_real_img:
                his_imgs = self.history_images
            else:
                left_imgs = self.act_model.init_state["left_imgs"][start_i:start_i+chunk_size+img_delay]
                right_imgs = self.act_model.init_state["right_imgs"][start_i:start_i+chunk_size+img_delay]
                his_imgs = [(left_imgs[i], right_imgs[i]) for i in range(len(left_imgs))]

        if state_delay > 0:
            h1_state = np.stack(his_state[:-state_delay], axis=0)
        else:
            h1_state = np.stack(his_state, axis=0)            
        if img_delay > 0:
            left_imgs, right_imgs = zip(*his_imgs[:-img_delay])
        else:
            left_imgs, right_imgs = zip(*his_imgs)

        onehot_task_id = self.skill_embedding

        # import pdb; pdb.set_trace()

        return h1_state, left_imgs, right_imgs, onehot_task_id

    def all_states_callback(self, msg):
        self.h1_state = np.array(msg.data)

        if self.h1_state.shape[0] == 53:
            # self.h1_state = self.h1_state.reshape(1, 36)
            self.h1_state[2:2+8] = self.h1_state[36:36+8]
            for i in range(10):
                self.h1_state[2+i] = self.h1_state[2+i] - self.skill_joint_offset_in_state[i]
            self.h1_state[36] = self.h1_state[-1]
            self.h1_state = self.h1_state[:37]

        if self.h1_state.shape[0] == 37:
            self.h1_state = self.h1_state[:36]

    def destroy_node(self):
        self.shm.close()
        self.shm.unlink()
        super().destroy_node()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop Node")
    parser.add_argument("-f", "--fake", action="store_true", help="Play fake video")
    parser.add_argument(
        "-n",
        "--ngrok",
        action="store_true",
        help="Use ngrok, open when using Quest, close when using VisionPro.",
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default="demo_videos/vr_record_robot_reach_out.pkl",
        help="fake video_path",
    )
    parser.add_argument("-r", "--record", action="store_true", help="Record data")
    parser.add_argument("--react", action="store_true", help="React mode, get zed image from shared memory")
    parser.add_argument("--iphone", type=str, choices=["none", "rgbd", "rgb", "rgb_lowd"], default="none", help="Use iphone camera")
    parser.add_argument("--scenario", type=str, default="dining", choices=["dining", "office", "0", "1"],
                        help="react scenario, 0: dining, 1: office")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-o", "--object", default='user', help="Test which object")
    parser.add_argument("-p", "--replay", nargs='+', type=int, default=[-1], help="Replay from sampled actions")
    parser.add_argument("--just-resize", action='store_true', help="used in the end-to-end model")
    parser.add_argument("--skill-cancel", default=False, action="store_true", help="Skill cancel mode")
    parser.add_argument("--info", type=str, default="", help="Extra info, save in scoreboard file name.")

    args = parser.parse_args()   
    
    if args.scenario in ["0", "1"]:
        args.scenario = "dining" if args.scenario == "0" else "office"

    try:
        
        # create mp data for record

        process_list = []
        
        [p.start() for p in process_list]

        rclpy.init()
        node = TeleopNode(
            model_path="../TeleVision/data/logs",
            data_path="../TeleVision/data/",
            react_mode=args.react,
            iphone=args.iphone,
            scenario=args.scenario,
            verbose=args.verbose,
            object=args.object,
            replay=args.replay,
            just_resize=args.just_resize,
            info=args.info,
            enable_skill_cancel=args.skill_cancel
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        [p.terminate() for p in process_list]
        [p.join() for p in process_list]

        node.get_logger().info("Keyboard Interrupt")
        node.destroy_node()
        rclpy.shutdown()
