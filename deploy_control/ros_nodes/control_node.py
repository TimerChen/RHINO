import os
import sys
import time
import subprocess
import argparse
from collections import deque

import numpy as np
from pytransform3d import rotations  # used in head_callback
import json


import multiprocessing as mp
from multiprocessing import shared_memory

import rclpy
import rclpy.qos
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy

from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray, MultiArrayDimension


from unitree_go.msg import (
    LowCmd,
    LowState,
    MotorCmd,
    MotorCmds,
    MotorState,
    MotorStates,
)
from unitree_sdk2py.utils.crc import CRC

from ros_nodes.wrists_controller import WristsController
from ros_nodes.head_controller import HeadController
from rclpy.node import Node

from motion_utils.consts import *
from ros_nodes.node_utils import ActionBuffer

QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
FPS = 30
class ControlNode(Node): # arms and fingers controller
    def __init__(self, args):
        super().__init__("control_node")
        self.args = args

        qos_profile = QOS_PROFILE

        # control_dt = 0.1
        control_dt = 0.02 * 30 / FPS
        # safe mode
        # max_joint_velocity = 3.0
        # active mode
        self.start_joint_velocity = 1.0
        self.safe_joint_velocity = 1.0
        self.running_joint_velocity = 6.0
        # 1.0 can raise up hands
        # 3.0 can only raise hands to shoulder

        self.final_pos = np.array([0.0, 0.0, 0.0, kPi_2, 0.0, 0.0, 0.0, kPi_2])
        self.init_pos_traj = np.array(
            [
                [0.0, 0.32, 0.0, kPi_2, 0.0, -0.32, 0.0, kPi_2],
                [0.0, 0.32, kPi_2, kPi_2, 0.0, -0.32, -kPi_2, kPi_2],
                [0.0, 0.32, kPi_2, -0.64, 0.0, -0.32, -kPi_2, -0.64],
                [0.0, 0.12, 0.0, -0.64, 0.0, -0.12, 0.0, -0.64],
                [0.0, 0.12, 0.0, -0.12, 0.0, -0.12, 0.0, -0.12]
            ]
        )
        self.check_pos = np.array([0.0, 0.0, -1.3, kPi_2, 0.0, 0.0, 1.3, kPi_2])
        if args.calib:
            with open("h1_assets/calib/manual_align.json", "r") as f:
                json_data = json.load(f)
                self.marker_align_pos = np.array(json_data["mean_list"])
        else:
            with open("h1_assets/calib/calib.json", "r") as f:
                json_data = json.load(f)
                self.offset = np.array(json_data["offset"])

        self.control_dt = control_dt
        self.max_joint_velocity = self.start_joint_velocity

        self.msg = LowCmd()  # unitree_go_msg_dds__LowCmd_()
        self.fingers_state_msg = MotorStates()

        self.fingers_cmd_msg = MotorCmds()
        for _ in range(12):
            self.fingers_cmd_msg.cmds.append(MotorCmd())

        self.all_states_msg = Float32MultiArray()
        self.all_states_msg.data = [0.0] * (52+1)
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="head_state", size=2))
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="arms_last_act", size=8))
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="wrists_state", size=2))
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="fingers_state", size=24))
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="arms_qpos", size=8))
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="arms_tau_est", size=8))
        self.all_states_msg.layout.dim.append(MultiArrayDimension(label="state_timestamp", size=1))
        self.all_states_msg.layout.data_offset = 0
        self.all_states_msg.data[52] = time.time()

        self.lowstate = None
        self.fingers_state = None
        self.crc = CRC()
        self.react_mode = args.react_mode
        self.manip_mode = args.manip_mode
        self.no_rt = args.no_rt

        self.target_pos = np.zeros(8)
        self.wrists_pos = np.zeros(2)

        self.safe_start_time = None
        self.safe_time = 3
        self.safe_lock = True

        # For safe strategy: avoid too large torque
        self.tau_queue = deque(maxlen=10)
        self.tau_lock = [0] * 8
        self.max_tau = [12] * 8
        self.min_tau = [-12] * 8
        self.min_tau[0], self.min_tau[4] = -15, -15
        self.max_tau[1], self.max_tau[5] = 15, 15
        # special rule for the elbow
        self.max_tau[3], self.max_tau[7] = 5, 5
        self.min_tau[3], self.min_tau[7] = -10, -10
        self.fast_mode = args.fast_mode
        self.head_only = args.head_only
        self.body_enable = {k: False for k in ["arm", "hand", "head", "wrists"]}

        if self.head_only:
            self.body_enable["head"] = True
        else:
            for k in self.body_enable.keys():
                self.body_enable[k] = True

        # start demon process for hand control
        # self.get_logger().info("Start hand control process...")

        os.system("find /dev -name 'ttyUSB*' -o -name 'ttyACM*' | xargs -I {} chmod 666 {}")
        # bashCommand = "bash /Share/scripts/control_hand.sh"
        
        # bashCommand = "/Code/h1_inspire_service/build/inspire_hand -s /dev/ttyUSB1"
        # bashCommand = "/Share/h1_inspire_service/build/inspire_hand"
        # self.fingers_control_process = subprocess.Popen(bashCommand.split(),stdout=subprocess.PIPE)
        # input("Press Enter to continue...")
        # output, error = process.communicate()

        self.qpos_deque = deque(maxlen=1000)
        self.state_deque = deque(maxlen=1000)

        # The msg sent to the robot
        self.arm_sdk_pub = self.create_publisher(LowCmd, "arm_sdk", qos_profile=qos_profile)
        self.arm_sdk_pub2 = self.create_publisher(Float32MultiArray, "arm_sdk2", qos_profile=qos_profile)
        # The msg received from the robot
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.lowstate_callback, qos_profile=qos_profile
        )

        # The msg sent to the inspire hand
        self.fingers_cmd_pub = self.create_publisher(MotorCmds, "/inspire/cmd", qos_profile=qos_profile)
        # The msg received from the inspire hand
        self.fingers_state_sub = self.create_subscription(
            MotorStates, "/inspire/state", self.fingers_state_callback, qos_profile=qos_profile
        )

        self.action_buffer = ActionBuffer(self.init_pos_traj[-1].tolist(), fps=20)
        # 3+2+10+24+4
        # self.full_action_buffer = ActionBuffer(fps=40, bit_size=43)
        print("create buffer")
        self.full_action_buffer = ActionBuffer(fps=40, bit_size=44)
        self.get_logger().info("Ready to init")
        self.init_hands()
        # exit()
        self.init_arms()
        self.init_head()
        self.init_wrists()
        self.init_hands()
        
        # The msg received from the react model
        self.react_sub = self.create_subscription(
            Float32MultiArray, "h1_pose", self.react_callback, qos_profile=qos_profile
        )

        # The msg received from the VR
        self.static_dict = {
            "arm_cnt": 0,
            "fingers_cnt": 0,
            "head_cnt": 0,
            "sync_cnt": 0,
            "arm_time": [],
            "fingers_time": [],
            "head_time": [],
            "sync_time": [],
        }
        split_mode = False
        if split_mode:
            self.arm_cmd_sub = self.create_subscription(
            Float32MultiArray, "arms_qpos", self.arm_qpos_callback, qos_profile=qos_profile
            )
            self.fingers_cmd_sub = self.create_subscription(
                Float32MultiArray, "fingers_qpos", self.fingers_qpos_callback, qos_profile=qos_profile
            )
            self.head_pose_sub = self.create_subscription(
                Float32MultiArray, "head_qpos", self.head_callback, qos_profile=qos_profile
            )
        else:
            self.all_rt_qpos_sub = self.create_subscription(
                Float32MultiArray, "all_rt_qpos", self.all_rt_qpos_callback, qos_profile=qos_profile
            )
        
        self.all_qpos_sub = self.create_subscription(
            Float32MultiArray, "all_qpos", self.all_qpos_callback, qos_profile=qos_profile
        )
        
        # From React model
        self.skill_start_sub = self.create_subscription(
            Float32MultiArray, "react/exec_skill", self.skill_signal_callback, qos_profile=qos_profile
        )
        # From Manip model
        self.skill_done_sub = self.create_subscription(
            Float32MultiArray, "manip/skill_done", self.skill_signal_callback, qos_profile=qos_profile
        )

        # The msg sent to vr_node (record)
        self.all_states_pub = self.create_publisher(Float32MultiArray, "all_states", qos_profile=qos_profile)

        self.sync_cnt = 0
        timer = self.create_timer(1, self.count_fps)
 
        self.timer = self.create_timer(1./FPS, self.sync_target_pos)
        self.pub_all_states_timer = self.create_timer(1./FPS, self.pub_all_states)
        self.check_start_timer = self.create_timer(0.1, self.check_start_pose)

        self.target_pos = self.init_pos_traj[-1].tolist()
        self.last_time = time.time()
        
    def skill_signal_callback(self, msg):
        """ Switch skills """
        if msg.data[0] > 0:
            self.get_logger().info("Start skill")
            self.manip_mode = True
            self.react_mode = False
            self.action_buffer.clear()
            self.full_action_buffer.clear()
        elif msg.data[0] == -1:
            self.get_logger().info("Finish skill")
            self.manip_mode = False
            self.react_mode = True
            # clear action buffer of react mode
            self.action_buffer.clear()
            self.full_action_buffer.clear()
        else:
            self.get_logger().info("Unknown signal")

    def log_motor_info(self):
        if not self.body_enable["arm"]:
            return
        states = self.lowstate.motor_state
        temps = [states[i].temperature for i in ArmJoints]
        tau = [states[i].tau_est for i in ArmJoints]
        motor_names = [i for i in ArmJointsName]
        self.get_logger().info(f"Motor: {motor_names}")
        self.get_logger().info(f"Motor T: {temps}")
        self.get_logger().info(f"Motor Tau: {tau}")

    def count_fps(self):
        cnt0 = self.static_dict["head_cnt"]
        cnt1 = self.static_dict["arm_cnt"]
        cnt2 = self.static_dict["fingers_cnt"]

        cnt3 = self.static_dict["sync_cnt"]
        self.get_logger().info(f"FPS: {self.sync_cnt} head {cnt0} arm {cnt1} fingers {cnt2}")
        t0 = self.static_dict["head_time"]
        t1 = self.static_dict["arm_time"]
        t2 = self.static_dict["fingers_time"]
        t3 = self.static_dict["sync_time"]
        # if len(t0) > 0:
        #     self.get_logger().info(f"Time head: {np.sum(t0)} {np.mean(t0)} {np.max(t0)}")
        
        # if len(t1) > 0:
        #     self.get_logger().info(f"Time arm: {np.sum(t1)} {np.mean(t1)} {np.max(t1)}")
        
        # if len(t2) > 0:
            # self.get_logger().info(f"Time fingers: {np.sum(t2)} {np.mean(t2)} {np.max(t2)}")
        
        # if len(t3) > 0:
            # self.get_logger().info(f"Time sync: {np.sum(t3)} {np.mean(t3)} {np.max(t3)}")

        # self.get_logger().info(f"FPS: {cnt3} head {cnt0} arm {cnt1} fingers {cnt2}")
        # self.get_logger().info(f"Time head: {np.sum(t0)} {np.sum(t1)} {np.sum(t2)} {np.sum(t3)}")
        self.get_logger().info(f"Tau Lock {self.tau_lock}")

        self.log_motor_info()

        self.sync_cnt = 0
        self.static_dict = {
            "arm_cnt": 0,
            "fingers_cnt": 0,
            "head_cnt": 0,
            "sync_cnt": 0,
            "arm_time": [],
            "fingers_time": [],
            "head_time": [],
            "sync_time": [],
        }
    # @property
    def max_joint_delta(self, dt=None):
        if dt is None:
            if self.fast_mode:
                dt = self.control_dt * 2
            else:
                dt = self.control_dt
        return self.max_joint_velocity * dt

    def _is_safe_pose(self, pose):
        return np.max(np.abs(pose)) < 0.1

    def check_start_pose(self):
        # check_start_timer is unused. To open it, comment next 2 lines.
        if not self.safe_lock:
            return
        if self.safe_start_time is None:
            self.safe_start_time = time.time()
            self.max_vrnt_velocity = self.safe_joint_velocity
            self.get_logger().info("[Info] Start safe count down....")

        progress_ratio = (time.time() - self.safe_start_time) / self.safe_time
        self.max_joint_velocity = progress_ratio * (self.running_joint_velocity - self.safe_joint_velocity) + self.safe_joint_velocity
        self.get_logger().info(f"[Info] Safe count down: {self.safe_time - (time.time() - self.safe_start_time)}")
        
        if time.time() - self.safe_start_time > self.safe_time:
            self.max_joint_velocity = self.running_joint_velocity
            self.get_logger().info("[Info] Safe count down finished....")
            self.safe_lock = False

    def init_arms(self):
        if not self.body_enable["arm"]:
            return
        self.terminate = False
        self.last_target_pos = None
        self.set_weight(1.0)
        # self.step(self.init_pos_traj[0])
        time.sleep(1)

        # check whether the robot is in the initial position
        current_jpos = self.get_arms_state()
        print("current_pos", current_jpos)

        if self.args.calib:
            input(f"""
    current arm jpos is {[f'{q:.4f}' for q in current_jpos]},
    will aligned to {[f'{q:.4f}' for q in self.marker_align_pos]},
    with offset {[f'{q:.4f}' for q in current_jpos - self.marker_align_pos]},
    press any key to ensure...
    """)
            
            self.offset = current_jpos - self.marker_align_pos
            with open("h1_assets/calib/calib.json", "w") as f:
                json.dump({"offset": self.offset.tolist()}, f)
        else:
            self.get_logger().info(f"offset {[f'{q:.4f}' for q in self.offset]}")

        if np.max(np.abs(current_jpos - self.check_pos)) > 0.3:
            print("ERROR: Robot is not in the initial position")
            # self.move_arm_to_init()
            self.set_weight(1.0)
            # self.send_msg()
            raise Exception("Robot is not in the initial position")
            self.terminate = True
        else:
            print("Robot is in the initial position")
            self.terminate = False

        # self.step(self.init_pos)
        # =============== uncomment it
        self.set_weight(1.0)
        self.move_arm_to_init()
        # self.init_wrists()
        time.sleep(1)


    def react_callback(self, msg):
        """Put action sequence into buffer"""
        msg_data = np.array(msg.data).reshape(-1, 8)
        for i in range(msg_data.shape[0]):
            self.action_buffer.put(msg_data[i])

    def sync_target_pos(self):
        """Sync target_pos with action_buffer"""
        # self.target_pos[:] = self.action_buffer
        self.static_dict["sync_cnt"] += 1
        now = time.time()
        self.sync_cnt += 1

        # if self.react_mode:
        #     target_pos = self.action_buffer.get_current_target()
        #     self.target_pos = target_pos
        #     wrists_pos = None
        # el
        if self.manip_mode or self.react_mode:
            all_pos = self.full_action_buffer.get_current_target()
            if False:
                # head
                head_pos = all_pos[0:2]
                # arm and wrists
                arm_pos = all_pos[2:12]
                target_pos = arm_pos[[0, 1, 2, 3, 5, 6, 7, 8]]
                wrists_pos = arm_pos[[4, 9]]
                # fingers
                fingers_pos = np.clip(all_pos[12:36], 0, 1)
                # print(f"Get: fingers{fingers_pos[:12].round(1)}")
            else:
                self.parse_instruction(all_pos)
                # print("Do all pos", all_pos)
            
            # self.active_head_callback(head_pos)
            # self.fingers_qpos_callback(fingers_pos)
            
            # self.get_logger().info(str(self.sync_cnt))
            # if self.sync_cnt % 1000 == 0:
                # self.save_queue()
            target_pos = self.target_pos
            wrists_pos = self.wrists_pos
        else:
            target_pos = self.target_pos
            wrists_pos = self.wrists_pos
        # self.get_logger().info(f"Waiting actions: {self.action_buffer.buffer.qsize()}")
        # self.get_logger().info('I read from array: "%s"' % self.target_pos)

        self.move_arm_to_pos(target_pos, wrists_pos)
        self.static_dict["sync_time"].append(time.time() - now)

    def parse_instruction(self, msg_data):
        if msg_data[0] > 0.: # and self.static_dict["sync_cnt"] % 10 == 0:
            self.head_callback(msg_data[3:3+2])
        # print(f"head_callback:{(time.time()-now):.4f}")
        # now = time.time()
        
        if len(msg_data) > 3+2+10+24:
            lr_mask = msg_data[3+2+10+24:3+2+10+24+4]
        else:
            lr_mask = [1]*4

        if msg_data[1] > 0.:
            self.arm_qpos_callback(msg_data[3+2:3+2+10], lr_mask[:2])
        # print(f"arms_qpos_callback:{(time.time()-now):.4f}")
        # now = time.time()

        if msg_data[2] > 0.:
            self.fingers_qpos_callback(msg_data[3+2+10:3+2+10+24], lr_mask[2:])
        

    def all_rt_qpos_callback(self, msg):
        """ receive all qpos """
        if self.no_rt:
            return
        msg_data = np.array(msg.data).reshape(-1)
        self.parse_instruction(msg_data)
        # TODO: deal with lr_mask
        # print(f"fingers_qpos_callback:{(time.time()-now):.4f}")
        # now = time.time()

    def all_qpos_callback(self, msg):
        """ receive all qpos """
        msg_data = np.array(msg.data).reshape(-1, 44)   
        for i in range(msg_data.shape[0]):
            self.full_action_buffer.put(msg_data[i], msg_data[i, -1])

    def arm_qpos_callback(self, msg, lr_mask=[1,1]):
        self.static_dict["arm_cnt"] += 1
        now = time.time()
        if not isinstance(msg, np.ndarray):
            target_pos = np.array(msg.data)
        else:
            target_pos = msg

        # self.get_logger().info(f"Receive VR message: {msg.data}")
        if lr_mask[0]:
            self.target_pos[:4] = target_pos[[0, 1, 2, 3]]
            self.wrists_pos[0] = target_pos[4]
        if lr_mask[1]:
            self.target_pos[4:] = target_pos[[5, 6, 7, 8]]
            self.wrists_pos[1] = target_pos[9]
        # self.target_pos[:] = target_pos[[0, 1, 2, 3, 5, 6, 7, 8]]
        # self.wrists_pos[:] = target_pos[[4, 9]]
        self.static_dict["arm_time"].append(time.time() - now)
        # self.target_pos[:] = target_pos
        # print("vr send msg", target_pos)

    def fingers_qpos_callback(self, msg, lr_mask=[1,1]):
        self.static_dict["fingers_cnt"] += 1
        now = time.time()
        if isinstance(msg, Float32MultiArray):
            qpos = np.array(msg.data)
        else:
            qpos = msg

        # extract
        qpos = qpos[[
            0,  1,  4,  6,  8,  10, 
            12, 13, 16, 18, 20, 22
            ]].reshape(2, 6)
        # remap
        qpos = qpos[
            :, [5, 4, 3, 2, 1, 0]
        ]

        
        qpos = np.clip(qpos, 0, 1)
        qpos = 1 - qpos
        # print(f"Set: left fingers{qpos[0].round(1)} at time {time.time()-self.last_time}")
        self.last_time = time.time()
        self.move_fingers_to_qpos(qpos[1], qpos[0], lr_mask)
        self.static_dict["fingers_time"].append(time.time() - now)

    def move_fingers_to_qpos(self, right_angles, left_angles, lr_mask=[1,1]):
        if not self.body_enable["hand"]:
            return
        # print("right", right_angles, left_angles)
        # self.get_logger().info(f"Move fingers to qpos: {right_angles}, {left_angles}")
        if lr_mask[1]:
            for i in range(6):
                self.fingers_cmd_msg.cmds[i].q = float(right_angles[i])

        if lr_mask[0]:
            for i in range(6):
                self.fingers_cmd_msg.cmds[i + 6].q = float(left_angles[i])

        self.fingers_cmd_pub.publish(self.fingers_cmd_msg)

    # def active_head_callback(self, msg):
    #     """Used in manip mode"""
    #     if isinstance(msg, Float64MultiArray):
    #         ypr = np.array(msg.data)
    #     else:
    #         ypr = msg
    #     # self.get_logger().info(f"[active_head_callback] head_yp: {ypr} ")
    #     self.qpos_deque.append([ypr[0], ypr[1], time.time()])
    #     self.head_controller.set_position_from_head_yp(ypr[:2])

    def smooth_head_move(self, ypr):
        v_max = np.pi / 2 / 75
        diff = ypr - self.head_last_pos
        diff = np.clip(diff, -v_max, v_max)
        ypr = self.head_last_pos + diff
        self.head_last_pos = ypr
        return ypr

    def head_callback(self, msg):
        """Used in teleop mode"""
        if not self.body_enable["head"]:
            return
        try:
            if isinstance(msg, Float32MultiArray):
                head_yp = msg.data
            else:
                head_yp = msg
            
            head_yp = self.smooth_head_move(head_yp)
            self.head_controller.set_position_from_head_yp(head_yp, human_offset=False)
        except Exception as e:
            print(e)
            self.get_logger().error("Get head quat and set command failed.")
            exit(-1)
            pass

        # self.static_dict["head_time"].append(time.time() - now)

    #    def timer_callback(self):
    #        self.target_pos[:] = self.shared_array[:]
    #        # self.get_logger().info('I read from array: "%s"' % self.target_pos)
    #        self.move_arm_to_pos(self.target_pos)

    def lowstate_callback(self, msg):
        self.lowstate = msg
        if not self.safe_lock:
            self.last_target_pos = np.array([msg.motor_state[i].q for i in ArmJoints])
        # self.get_logger().info('lowstate heard: "%s"' % self.get_arms_state())

    def fingers_state_callback(self, msg):
        self.fingers_state = msg

    def move_wrists_to_pos(self, wrists_pos):
        """Convert vr wrist pos to motor pos
        vr: [-3 ~ 3] h1: [0 ~ 5.75]
        map: vr[-2~3]-> h1[0~5]
        """
        # self.get_logger().info('VR wrists to pos: "%s"' % wrists_pos)
        # wrists_pos[1] = -wrists_pos[1]
        # wrists_pos = (wrists_pos + 2)
        offset0 = 2.8
        wrists_pos[0] = wrists_pos[0] + offset0
        wrists_pos[1] = wrists_pos[1] - offset0

        self.wrists_controller.ctrl(wrists_pos)
        # debug
        # self.get_logger().info('Move wrists to pos: "%s"' % wrists_pos)
        # self.get_logger().info(f"Wrist State: {self.wrists_controller.get_state()}")

    def move_arm_to_pos(self, target_pos, wrists_pos):
        if not self.body_enable["arm"]:
            return
        
        self.move_wrists_to_pos(wrists_pos.copy())

        # current_time = time.time()
        # if current_time - self.last_time < 0.001:
        #    time.sleep(0.001 - current_time + self.last_time)
        # self.last_time = time.time()
        # return

        target_pos = [
            np.clip(a, ArmJointLowerBounds[j], ArmJointUpperBounds[j])
            for j, a in zip(ArmJoints, target_pos)
        ]

        current_jpos_des = self.get_arms_state()
        # for i in range(1000):
        # if np.max(np.abs(current_jpos_des - target_pos)) < 0.01:
        #     break

        target_pos = self.move_from_to(current_jpos_des, target_pos)
        # for j in range(len(target_pos)):
        #     current_jpos_des[j] += np.clip(
        #         target_pos[j] - current_jpos_des[j],
        #         -self.max_joint_delta,
        #     )

        # self.step(current_jpos_des)
        # NOTE: here may be a bug, the last joint may not reach the target
        if False:
            for j in range(len(ArmJoints)):
                self.lowstate.motor_state[ArmJoints[j]].q = target_pos[j]

    def check_tau(self):
        tau_ests = len(ArmJoints) * [0.0]
        for j in range(len(ArmJoints)):
            # tau estimation
            tau_ests[j] = self.lowstate.motor_state[ArmJoints[j]].tau_est
        print("tau est", tau_ests)

    def init_wrists(self):
        if not self.body_enable["wrists"]:
            return
        self.wrists_controller = WristsController("/dev/ttyACM0")
        self.wrists_controller.init_motors()

    def init_head(self):
        if not self.body_enable["head"]:
            return
        self.head_controller = HeadController("/dev/ttyUSB0")
        # The other msg received from the VR (teleop_head_node.py)
        self.head_controller.set_position_from_head_yp(np.array([0.3, 0.0]), old_mode=True) # TODO: check the joint angle.
        self.head_last_pos = np.array([0.3, 0.0])

    def init_hands(self):
        if not self.body_enable["hand"]:
            return
        self.move_fingers_to_qpos([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        time.sleep(1)
        self.move_fingers_to_qpos([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        time.sleep(1)
        self.move_fingers_to_qpos([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
        time.sleep(1)

    def move_arm_to_init(self):
        for init_pos in self.init_pos_traj:
            self.get_logger().info(f"[move_arm_to_init] move to {init_pos}")
            target_pos = init_pos
            total_time = 2000*0.005
            for i in range(2000):
                current_jpos_des = self.get_arms_state()
                if np.max(np.abs(current_jpos_des - target_pos)) < 0.1:
                    break
                self.move_from_to(current_jpos_des, target_pos, dt=self.control_dt)
                rclpy.spin_once(self)
                time.sleep(0.005)
                # time.sleep(0.033333)
            time.sleep(0.005)

    def move_from_to(self, current_jpos_des, target_pos, dt=None):
        delta = self.max_joint_delta(dt=dt)
        target_pos = target_pos + self.offset
        if self.last_target_pos is None:
            self.last_target_pos = current_jpos_des
        for j in range(len(target_pos)):
            self.last_target_pos[j] = self.last_target_pos[j] + np.clip(
            # target_pos_[j] = current_jpos_des[j] + np.clip(
                target_pos[j] - self.last_target_pos[j],
                -delta,
                delta,
            )
            # current_jpos_des[j] = current_jpos_des[j] + np.clip(
            #     target_pos[j] - current_jpos_des[j],
            #     -self.max_joint_delta,
            #     self.max_joint_delta,
            # )
        # print("target pos", target_pos_, "current pos", current_jpos_des)

        self.step(self.last_target_pos)
        # if not self.safe_lock:
        #     self.last_target_pos = current_jpos_des

        # self.step(current_jpos_des)
        return self.last_target_pos

    def pub_all_states(self):
        # only publish if all parts are enabled
        if not all(self.body_enable.values()):
            return
        self.get_all_states()
        self.all_states_pub.publish(self.all_states_msg)

    def get_all_states(self):
        """ Get all states 
        0~1: head
        2~9: arms old qpos (smoothed last action)
        10~11: wrists
        12~35: fingers
        36~43: arms current qpos
        44~51: arms current tau
        52: timestamp
        """
        self.all_states_msg.data[52] = time.time()
        head_state = self.get_head_state()
        # self.get_logger().info([head_state], head_state)
        for i in range(2):
            self.all_states_msg.data[i] = head_state[i]
        arms_state = self.get_arms_state()
        for i in range(8):
            self.all_states_msg.data[i + 2] = arms_state[i] - self.offset[i]
        wrists_state = self.get_wrists_state()
        for i in range(2):
            self.all_states_msg.data[i + 10] = wrists_state[i]
        fingers_state = self.get_fingers_state()
        for i in range(24):
            self.all_states_msg.data[i + 12] = fingers_state[i]

        arms_state = self.get_arms_state(old_mode=False)
        for i in range(8):
            self.all_states_msg.data[i + 36] = arms_state[i]

        arms_state = self.get_arms_state(stype='tau')
        for i in range(8):
            self.all_states_msg.data[i + 44] = arms_state[i]
    
    def get_head_state(self):
        if not self.body_enable["head"]:
            return
        head_yp_state= self.head_controller.get_position_as_head_yp()
        # self.get_logger().info(f"[get_head_state] head_yp: {head_yp_state} ")
        self.state_deque.append([head_yp_state[0], head_yp_state[1], time.time()])
        return head_yp_state

    def get_wrists_state(self):
        if not self.body_enable["wrists"]:
            return
        wrists_state = self.wrists_controller.get_state()['q']
        return wrists_state
    
    def get_fingers_state(self):
        if not self.body_enable["hand"]:
            return
        while True:
            if self.fingers_state is not None:
                msg = self.fingers_state
                states = msg.states
                qs = [states[i].q for i in range(12)]
                fs = [states[i].tau_est for i in range(12)]
                ts = [states[i].temperature for i in range(12)]
                return np.array(qs + fs)
            else:
                rclpy.spin_once(self)
                time.sleep(0.01)

    def get_arms_state(self, old_mode=True, stype=None):
        if not self.body_enable["arm"]:
            return
        if stype is not None:
            old_mode = False
        while True:
            if self.lowstate is not None:
                if old_mode and self.last_target_pos is not None:
                    return self.last_target_pos
                else:
                    msg = self.lowstate
                    motor_state = msg.motor_state
                    if stype == 'tau':
                        return np.array([motor_state[ArmJoints[j]].tau_est for j in range(len(ArmJoints))])
                    return np.array([motor_state[ArmJoints[j]].q for j in range(len(ArmJoints))])
            else:
                rclpy.spin_once(self)
                time.sleep(0.01)


    def save_queue(self):
        save_dir = "figs"
        np.save(os.path.join(save_dir,"qpos.npy"), np.array(self.qpos_deque))
        np.save(os.path.join(save_dir,"state.npy"), np.array(self.state_deque))
        self.get_logger().info("Save queue")


    def set_weight(self, weight):
        self.msg.motor_cmd[JointIndex["kNotUsedJoint"]].q = weight

    def _safe_strategy(self, action):
        if self.safe_lock:
            return action
        self.tau_queue.append([self.lowstate.motor_state[j].tau_est for j in ArmJoints])
        mean_tau = np.mean(np.array(self.tau_queue), axis=0)
        safe_action = self.get_arms_state(old_mode=False)

        for j in range(len(ArmJoints)):
            if self.tau_lock[j] != 0:
                # try to unlock
                if (action[j] - safe_action[j]) * self.tau_lock[j] < -0.05:
                    self.tau_lock[j] = 0

            if self.tau_lock[j] == 0:
                # try to lock
                if mean_tau[j] > self.max_tau[j]:
                    self.tau_lock[j] = 1
                    # # NOTE: special rule for the elbow, if the elbow is locked, the shoulder-pitch should be locked too
                    # if j == 3:
                    #     self.tau_lock[0] = 1
                    # elif j == 7:
                    #     self.tau_lock[4] = 1

                elif mean_tau[j] < self.min_tau[j]:
                    self.tau_lock[j] = -1

            if self.tau_lock[j] != 0:
                # if still locked, use safe action
                action[j] = safe_action[j]

        return action

    # safe mode: kp, kd 60, 1.5
    # active mode: kp, kd: 100, 6
    def step(
        self, action, dq=0.0, kp=120.0, kd=6.0, tau_ff=0.0
    ):  # action = target_joints
        # set control joints
        # print("action", action, type(action))
        action = self._safe_strategy(action)
        for j in range(len(action)):
            # self.msg.motor_cmd[ArmJoints[j]].mode=0
            self.msg.motor_cmd[ArmJoints[j]].q = action[j]
            self.msg.motor_cmd[ArmJoints[j]].dq = dq
            self.msg.motor_cmd[ArmJoints[j]].kp = kp
            self.msg.motor_cmd[ArmJoints[j]].kd = kd
            self.msg.motor_cmd[ArmJoints[j]].tau = tau_ff

        # NOTE: for debug latency!
        simple_arm_sdk = Float32MultiArray()
        simple_arm_sdk.data = action.tolist()
        self.arm_sdk_pub2.publish(simple_arm_sdk)

        self.send_msg()

    def send_msg(self):
        # self.msg.crc = self.crc.Crc(self.msg)
        if not self.terminate:
            # self.get_logger().info(
            #    "I am going to publish msg %s"
            #    % [self.msg.motor_cmd[ArmJoints[j]].q for j in range(len(ArmJoints))]
            # )
            self.arm_sdk_pub.publish(self.msg)
            pass
        else:
            print("ERROR: Robot is terminated", self.get_arms_state())

    def destroy_node(self):
        # self.wrists_controller.set_fixed_target([kPi_2, -kPi_2])
        # terminate hand control process
        # self.fingers_control_process.terminate()
        # self.fingers_control_process.wait()
        super().destroy_node()



def main():
    parser = argparse.ArgumentParser(description="Control Node")
    parser.add_argument("-m", "--manip-mode", action="store_true", help="Use teleoperation mode to control the robot")
    parser.add_argument("-r", "--react-mode", action="store_true", help="Use react model to control the robot")
    parser.add_argument("-f", "--fast-mode", action="store_true", help="Use fast mode")
    parser.add_argument("--head-only", action="store_true", help="Use head")
    parser.add_argument("--no-rt", action="store_true", help="No real time command")
    parser.add_argument("--fps60", action="store_true", help="Use 60 fps")
    parser.add_argument("--calib", action="store_true", help="Calibration mode")
    # parser.add_argument("-s", "--strong-thumb", action="store_true", help="Use strong thumb")

    args = parser.parse_args()

    if args.fps60:
        global FPS
        FPS = 60

    rclpy.init()
    print("Start control node")
    node = ControlNode(args)

    input("Press any key to start control...")
    try:
        # while True:
        rclpy.spin(node)

    except Exception as e:  # KeyboardInterrupt:
        # node.move_arm_to_init()
        print(e)
        node.destroy_node()

        rclpy.shutdown()
        # shm.close()
        raise e



if __name__ == "__main__":
    main()
