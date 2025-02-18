import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32MultiArray, Int32, ByteMultiArray
import numpy as np
import math
import torch
import os
from collections import deque
from scipy.spatial.transform import Rotation as R
os.sys.path.append(os.getcwd())
from motion_controllers.pin_fk import PinFk

# Topic names
HUMAN_HAND_TOPIC = 'hand_position' # ReActGen/zed_react_node?
ROBOT_JOINTS_TOPIC = 'all_states' # control_node? 
UNSAFE_NOTIFY_TOPIC = 'unsafe_notify' # safe_node
ZED_POSE_TOPIC = 'zed_pose'
# URDF_FILE = 'h1_assets/urdf/h1_with_hand_teleop.urdf'
ZED_CALIB_FILE = "h1_assets/head_state_to_zed_pose.npy"

# Safety threshold in meters
SAFE_DISTANCE = 0.1

class ArmSafeNode(Node):
    def __init__(self):
        super().__init__('arm_safe_node')
        
        # Configure QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # Store latest received data
        self.human_pos = None
        self.robot_joints = None
        
        # Load robot model using pink
        self.solver = PinFk(viz=True)
        self.point_interested = {
            # "waist_joint": 1,
            ("left_shoulder_pitch_joint", 'universe'): [0.0],
            ("left_shoulder_yaw_joint", "left_shoulder_pitch_joint"):[0.0],
            ("left_elbow_joint", "left_shoulder_yaw_joint"): [0.0, 0.5],
            ("left_hand_joint", "left_elbow_joint"): [-0.5, 0.0, 0.5],
            ("right_shoulder_pitch_joint", 'universe'): [0.0],
            ("right_shoulder_yaw_joint", "right_shoulder_pitch_joint"):[0.0],
            ("right_elbow_joint", "right_shoulder_yaw_joint"): [0.0, 0.5],
            ("right_hand_joint", "right_elbow_joint"): [-0.33, 0.0, 0.5],
        }  # key: value -- there are {value} points to calculate collider between this point(within) and its parent link(without)
        
        # self.get_zedm_frame_zero_from_solver()
        self.zedm_frame_offset = np.array([0.0, 0.0, 0.60])
        self.head_state_array = np.zeros(2)

        self.human_hand = np.zeros((2, 3))
        self.human_hand_in_zedm_frame = np.zeros((2, 3))
        
        # load zed calibration table, (num_keys, 8): 
        # head_state: [:, 0:2], zed_rot: [:, 2:5], zed_trans: [:, 5:8]
        self.head_state_to_zed_pose_table = np.load(ZED_CALIB_FILE)

        # Subscribe to human hand position
        self.human_hand_sub = self.create_subscription(
            Float32MultiArray,
            HUMAN_HAND_TOPIC,
            self.human_hand_callback,
            qos_profile,
        )
        
        # Subscribe to robot joint angles
        self.robot_joints_sub = self.create_subscription(
            Float32MultiArray, 
            ROBOT_JOINTS_TOPIC,
            self.robot_joints_callback,
            qos_profile,
        )
        
        self.zed_pose_sub = self.create_subscription(
            Float32MultiArray,
            ZED_POSE_TOPIC,
            self.zed_pose_callback,
            qos_profile,
        )

        # Publisher for safety status
        self.safe_pub = self.create_publisher(
            Int32,
            UNSAFE_NOTIFY_TOPIC,
            qos_profile,
        )

        self.zed_pose_deque = deque(maxlen=20)
        # Create timer for periodic safety checks
        self.timer = self.create_timer(1./30., self.safety_check)

    def get_zedm_frame_zero_from_solver(self):
        ### DESPRACTED ###
        """Get the zero point of zedm frame from solver"""
        zero_qpos = np.zeros(38)
        zero_qpos[2] = np.pi # head joint bias
        zero_qpos[3] = np.pi/2 # head joint bias
        self.zedm_frame_offset = self.solver.get_zedm_frame_zero(zero_qpos)
        

    def zed_pose_callback(self, msg):
        data = np.array(msg.data)
        # read translation and rotation
        translation = data[3:]
        rotation = R.from_euler('xyz', data[:3]).as_matrix()

        # # Examine the drift of zed pose
        # default_rotation_new = np.array([0.1039067,  0.63599235, 0.05336103])
        # default_translation_new = np.array([-0.07394909, -0.24354336,  0.18538475])
        # default_rotation = np.array([0.03979955, 0.63786495, 0.20041607])
        # default_translation = np.array([-0.07663903, -0.23360081,  0.18711878])
        # self.get_logger().info(f'[zed_pose_callback] rotation_drift: {data[:3]-default_rotation_new}, translation_drift: {data[3:]-default_translation_new}')
        # self.get_logger().info(f'[zed_pose_callback] rotation_drift: {data[:3]-default_rotation}, translation_drift: {data[3:]-default_translation}')
        # self.get_logger().info(f'[zed_pose_callback] rotation: {data[:3]}, translation: {data[3:]}')

        self.zed_pose_deque.append((rotation, translation))
        
        
        # get frame at t, t-5, ..., t-5k
        useful_index = np.arange(-1, -len(self.zed_pose_deque), -5)
        
        # meshcat draw frame
        for i, idx in enumerate(useful_index):
            rot, trans = self.zed_pose_deque[idx]
            trans = trans + self.zedm_frame_offset
            # pose = np.eye(4)
            # pose[:3,:3] = rot 
            # pose[:3,3] = trans + self.zedm_frame_zero
            
            self.solver.mark_frame(f"zed_pose_frame_{i}", frame_position=trans, frame_rotation=rot, opacity=0.5*(i+1)/len(useful_index))

    
    def human_hand_callback(self, msg):
        # self.get_logger().info("human hand callback: {}".format(msg.data))
        
        human_hand_in_zedm_frame = np.array(msg.data).reshape(-1, 3)
        # if point is not [0,0,0], then it is valid
        valid_points_idx = np.where(np.linalg.norm(human_hand_in_zedm_frame, axis=1) > 0)[0]
        # import ipdb; ipdb.set_trace()

        if len(valid_points_idx) > 0:
            self.human_hand_in_zedm_frame = human_hand_in_zedm_frame[valid_points_idx, :]

    # format of all_states
    # self.all_states_msg = Float32MultiArray()
    # self.all_states_msg.data = [0.0] * (52+1)
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="head_state", size=2)) ------ [0:2]
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="arms_last_act", size=8)) --- [2:10]
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="wrists_state", size=2)) ---- [10:12]
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="fingers_state", size=24)) -- [12:36]
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="arms_qpos", size=8)) ------- [36:44]
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="arms_tau_est", size=8)) ---- [44:52]
    # self.all_states_msg.layout.dim.append(MultiArrayDimension(label="state_timestamp", size=1)) - [52:53]
    # self.all_states_msg.layout.data_offset = 0
    # self.all_states_msg.data[52] = time.time()

    def robot_joints_callback(self, msg):
        """Callback for robot joint angle updates"""
        # print("robot joints callback: ", msg.data)
        # import pdb; pdb.set_trace()
        self.robot_joints = np.concatenate((
            [0], # universe
            [0.0, msg.data[0]+np.pi, msg.data[1]+np.pi/2],
            msg.data[36:40],
            [msg.data[10]+np.pi],
            np.zeros(12),# 1 - np.array(msg.data[24:36]),
            msg.data[40:44],
            [msg.data[11]+np.pi],
            np.zeros(12),# 1 - np.array(msg.data[12:24]),
        ))
        
        self.head_state_array[:] = msg.data[0:2]

    # joint in h1_with_hand_active_head.urdf
    # 0 universe
    # 1 active_head_base_joint
    # 2 active_head_center_joint
    # 3 active_head_neck_joint
    # 4 left_shoulder_pitch_joint
    # 5 left_shoulder_roll_joint
    # 6 left_shoulder_yaw_joint
    # 7 left_elbow_joint
    # 8 left_hand_joint
    # 9 L_index_proximal_joint
    # 10 L_index_intermediate_joint
    # 11 L_middle_proximal_joint
    # 12 L_middle_intermediate_joint
    # 13 L_pinky_proximal_joint
    # 14 L_pinky_intermediate_joint
    # 15 L_ring_proximal_joint
    # 16 L_ring_intermediate_joint
    # 17 L_thumb_proximal_yaw_joint
    # 18 L_thumb_proximal_pitch_joint
    # 19 L_thumb_intermediate_joint
    # 20 L_thumb_distal_joint
    # 21 right_shoulder_pitch_joint
    # 22 right_shoulder_roll_joint
    # 23 right_shoulder_yaw_joint
    # 24 right_elbow_joint
    # 25 right_hand_joint
    # 26 R_index_proximal_joint
    # 27 R_index_intermediate_joint
    # 28 R_middle_proximal_joint
    # 29 R_middle_intermediate_joint
    # 30 R_pinky_proximal_joint
    # 31 R_pinky_intermediate_joint
    # 32 R_ring_proximal_joint
    # 33 R_ring_intermediate_joint
    # 34 R_thumb_proximal_yaw_joint
    # 35 R_thumb_proximal_pitch_joint
    # 36 R_thumb_intermediate_joint
    # 37 R_thumb_distal_joint

    def get_robot_points(self):
        """Calculate key points along robot arm using forward kinematics
        Returns:
            list: List of 3D points representing robot arm segments
            None: If joint data not available
        """
        if self.robot_joints is None:
            return None
            
        points = []
        
        # Get joint positions using pink forward kinematics
        joint_positions = self.solver.fk(self.robot_joints, self.point_interested.keys())
        
        for (point_name, parent_point_name), interpolate_scale_list in self.point_interested.items():
            # import ipdb; ipdb.set_trace()
            link_pos, parent_link_pos = joint_positions[point_name]
            # for i in range(interpolate_num):
                # points.append(link_pos + (parent_link_pos - link_pos) * i / interpolate_num)

            for _scale in interpolate_scale_list:
                points.append(link_pos + (parent_link_pos - link_pos) * _scale)
        
        self.solver.mark_points({f'point_{i}': points[i] for i in range(len(points))}, color=0x00ffff, sphere_radius=SAFE_DISTANCE/2)
        # self.solver.mark_points({f'zedm_frame_zero': self.zedm_frame_zero}, color=0xff00ff, sphere_radius=SAFE_DISTANCE/2)
        # only draw joint_positions
        # self.solver.mark_points({key: joint_positions[key][1] for key in joint_positions.keys()}, color=0x0000ff)
        return points

    def safety_check(self):
        """Periodic check for unsafe distances between human and robot"""
        if self.human_hand is None or self.robot_joints is None:
            return
            
        robot_points = self.get_robot_points()
        if robot_points is None:
            return
        
        # The followiing two are under zedm frame, zed_pose_from_zed: get from zed msg, zed_pose_from_head: get from zed calibration table.
        zed_pose_from_zed = self.zed_pose_deque[-1][1] # element: (rotation, translation)
        zed_pose_from_head = self.head_state_to_zed_pose_table[
            np.argmin(np.sum(np.abs(self.head_state_to_zed_pose_table[:, 0:2] - self.head_state_array), axis=1))
        ][5:8]
        
        self.solver.mark_frame("zedm_frame_from_head", frame_position=zed_pose_from_head+self.zedm_frame_offset, frame_rotation=self.zed_pose_deque[-1][0], opacity=1.0)

        # import ipdb; ipdb.set_trace()
        # calculate the exact position of human hand in world frame
        self.human_hand = self.human_hand_in_zedm_frame - zed_pose_from_zed + zed_pose_from_head + self.zedm_frame_offset
        
        
        # calculate the distance between human hand and robot points
        dist = np.min(np.linalg.norm(self.human_hand[:, np.newaxis] - np.array(robot_points), axis=2))
        
        if dist < SAFE_DISTANCE:
            # Publish unsafe signal
            msg = Int32()
            msg.data = 1
            marker_color = 0xff0000
            self.safe_pub.publish(msg)
            self.get_logger().info("[safety_check] Unsafe distance detected")
            
        else:
            # Safe condition
            msg = Int32()
            msg.data = 0
            marker_color = 0x00ff00
            self.get_logger().info("[safety_check] Safe distance")

        self.safe_pub.publish(msg)\
        
        self.solver.mark_points({f'unsafe_point_{i}': self.human_hand[i] for i in range(len(self.human_hand))}, color=marker_color, sphere_radius=SAFE_DISTANCE/2)
            

def main(args=None):
    rclpy.init(args=args)
    node = ArmSafeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
