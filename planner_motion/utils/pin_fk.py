import pinocchio as pin
import numpy as np
import time
import hppfcl

import qpsolvers
from pytransform3d.transformations import (
    transform_from_pq,
    pq_from_transform,
)

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask
from pink.visualization import start_meshcat_visualizer
import meshcat
import meshcat_shapes


class LPConstrainedSE3Filter:
    def __init__(self, alpha, dt=1 / 60):
        self.alpha = alpha
        self.dt = dt
        self.is_init = False

        self.prev_pos = None
        self.prev_vel = None

        self.max_vel = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
        self.max_acc = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])

    def next(self, target):
        if not self.is_init:
            self.prev_pos = pin.SE3(target)
            self.prev_vel = np.zeros(6)
            self.prev_acc = np.zeros(6)
            self.is_init = True
            return np.array(self.prev_pos)

        ip_pos = pin.SE3.Interpolate(self.prev_pos, pin.SE3(target), self.alpha)
        ip_vel = pin.log(ip_pos.actInv(self.prev_pos)).vector / self.dt
        ip_acc = (ip_vel - self.prev_vel) / self.dt

        acc = np.clip(ip_acc, -self.max_acc, self.max_acc)
        vel = np.clip(self.prev_vel + acc * self.dt, -self.max_vel, self.max_vel)
        pos = self.prev_pos * (
            ~pin.exp(vel * self.dt)
        )  # Caution! * means matrix multiplication in pinocchio

        self.prev_pos = pos
        self.prev_vel = vel

        return np.array(self.prev_pos)

    def reset(self):
        self.y = None
        self.is_init = False


class PinFk:
    def __init__(
        self,
        fake_wrist_dof: bool = True,
        eps: float = 0.2,
        MAX_TIME: float = 0.01,
        DT: float = 0.1,
        damp: float = 1e-6,
        in_lp_alpha: float = 0.9,
        orientation_cost: float = 0.5,
        viz: bool = False,
    ):
        self.fake_wrist_dof = fake_wrist_dof
        wbc_robot = pin.RobotWrapper.BuildFromURDF(
            filename="./h1_assets/urdf/h1_wrist_pink_v819_active_head.urdf",
            package_dirs=["./h1_assets/urdf"],
            root_joint=None,
        )

        locked_joint_id = [
            int(wbc_robot.model.getJointId(joint_name))
            for joint_name in [
                "right_ankle_joint",
                "torso_joint",
                "left_ankle_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_hip_yaw_joint",
                "left_hip_roll_joint",
                "left_hip_pitch_joint",
                "right_hip_yaw_joint",
                "right_hip_roll_joint",
                "right_hip_pitch_joint",
                "pelvis",
            ]
        ]
        self.robot = wbc_robot.buildReducedRobot(locked_joint_id)
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.robot.q0
        )
        self.zed_link_name = "zedm_left_camera_frame"
        self.zed_point_bias = np.array([0.2, 0.0, 0.2])
        self.world_point_bias = np.array([0.0, 0.0, 0.0])

        if viz:
            self.viz = start_meshcat_visualizer(wbc_robot)
            self.viewer = self.viz.viewer
            
            # Create geometry model for visualization
            self.geom_model = pin.GeometryModel()
            self.marker_geom = hppfcl.Sphere(1)

            meshcat_shapes.frame(self.viewer[self.zed_link_name], opacity=0.5)
            
        else:
            self.viz = None
            self.viewer = None
        
        
        self.right_wrist_task = FrameTask(
            "right_hand_yaw_link" if fake_wrist_dof else "right_hand_link",
            position_cost=1.0,
            orientation_cost=orientation_cost,
        )
        self.left_wrist_task = FrameTask(
            "left_hand_yaw_link" if fake_wrist_dof else "left_hand_link",
            position_cost=1.0,
            orientation_cost=orientation_cost,
        )
        posture_task = PostureTask(
            cost=1e-1,  # [cost] / [rad]
        )
        self.solver = qpsolvers.available_solvers[0]
        self.dt = 0.01
        self.damp = damp

        self.tasks = [
            self.left_wrist_task,
            self.right_wrist_task,
            posture_task,
        ]

        # for idx, name in enumerate(self.robot.model.names):
        #     print(idx, name)

        # Task target specifications
        posture_task.set_target_from_configuration(self.configuration)
        # q = self.configuration.q.copy()
        # q[3] = 0.32  # left shoulder roll
        # q[20] = -0.32  # right shoulder roll
        # posture_task.set_target(q)

        self.left_wrist_filter = LPConstrainedSE3Filter(in_lp_alpha)
        self.right_wrist_filter = LPConstrainedSE3Filter(in_lp_alpha)

        self.finetune_dict = {
            "active_head_base_joint": 0.05,
            "active_head_center_joint": 0.03,
            "active_head_neck_joint": -0.09,
            "left_shoulder_pitch_joint": 0.1,
            "left_shoulder_roll_joint": 0.1,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.05,
            "right_shoulder_pitch_joint": 0.1,
            "right_shoulder_roll_joint": -0.1,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.05,
        }

    def fk(
        self,
        qpos,
        interested_points,
    ):
        """
        qpos: joint angles
        interested_points: interested points to calculate
        return: point_positions: {point_name: [x, y, z]}, positions on joint points.
        """
        joint_values = self._fk(qpos)
        
        # get points from robot oMf(=origin frame coordinate)
        point_positions = {}
        # import ipdb; ipdb.set_trace()
        for point_name in interested_points:
            frame_id = self.robot.model.getFrameId(point_name)
            point_position = self.robot.data.oMf[frame_id].translation
            point_positions[point_name] = point_position.copy()
        if self.viz:
            print("Updating robot display")
            # Update robot display
            pin.forwardKinematics(self.robot.model, self.robot.data, joint_values)
            self.viz.display(np.concatenate((np.zeros(11,), joint_values), axis=0)) # 11 for fixed joints.
                
        return point_positions
    
    def _fk(self, qpos):
        # update robot by qpos
        # self.configuration.q = qpos
        joint_values = np.zeros(self.robot.model.nv + 1)
        # import ipdb; ipdb.set_trace()
        for q, joint_name in zip(qpos, self.robot.model.names):
            finetune_delta = self.finetune_dict.get(joint_name, 0)
            q = q + finetune_delta

            joint_id = self.robot.model.getJointId(joint_name)
            # print(joint_name, joint_id, q)
            joint_values[joint_id] = q
        joint_values = joint_values[1:]
        pin.forwardKinematics(self.robot.model, self.robot.data, joint_values)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        return joint_values

    def get_zedm_frame_zero(self, zero_qpos):
        ### DESPRACTED ###
        zero_joint_values = self._fk(zero_qpos)
        zed_frame_id = self.robot.model.getFrameId(self.zed_link_name)
        zedm_frame_zero = self.robot.data.oMf[zed_frame_id].translation

        return zedm_frame_zero


    def transform_from_zedm_frame(self, points_in_zedm_frame):
        # points is in zedm frame, transform them to robot frame.
        # zedm frame: x: right, y: down, z: forward
        # robot frame: x: forward, y: left, z: up
        # zedm frame to robot frame: rotate 90 degree around x axis, then rotate 90 degree around z axis.
        # R = Rz(90) * Rx(90)
            # Draw the frame of zedm_camera_center
        zedm_camera_center_frame_id = self.robot.model.getFrameId(self.zed_link_name)
        zedm_camera_center_position = self.robot.data.oMf[zedm_camera_center_frame_id].translation
        zedm_camera_center_rotation = self.robot.data.oMf[zedm_camera_center_frame_id].rotation

        # transform points in zedm frame to robot frame, using zedm_camera_center_frame
        points_in_world_frame = np.zeros_like(points_in_zedm_frame)
        for i in range(points_in_zedm_frame.shape[0]):
            point_in_zedm_frame = points_in_zedm_frame[i, :] + self.zed_point_bias
            point_in_zedm_frame_3d = np.array([point_in_zedm_frame[0], point_in_zedm_frame[1], point_in_zedm_frame[2]])
            points_in_world_frame[i, :] = self.robot.data.oMf[zedm_camera_center_frame_id].act(point_in_zedm_frame_3d)
            points_in_world_frame[i, :] += self.world_point_bias

        if self.viz:
            self.viewer[self.zed_link_name].set_transform(
                self.robot.data.oMf[zedm_camera_center_frame_id].np
            )
            # self.mark_points({"zedm_camera_center": zedm_camera_center_position}, color=0x00ffff)
            # Draw the frame of zedm_camera_center
            # self.viz.addFrame("zedm_camera_center", zedm_camera_center_position, zedm_camera_center_rotation)

        # R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        # T = np.eye(4)
        # T[:3, :3] = R
        # points_in_robot_frame = np.dot(T, points_in_zedm_frame)
        # return points_in_robot_frame
        return points_in_world_frame
    
    def mark_frame(self, frame_name, frame_position, frame_rotation, color=0xff0000, opacity=0.5):
        # if not self.viewer.has(frame_name):
        #     self.viewer.addFrame(frame_name, frame_position, frame_rotation)
        # else:
        self.viewer[frame_name].set_transform(meshcat.transformations.translation_matrix(frame_position))
        meshcat_shapes.frame(self.viewer[frame_name], opacity=opacity)
        
    def mark_points(self, point_positions, color=0xff0000, sphere_radius=None, cube_size=None, opacity=0.5):
        
        # # ================= Example =================
        # # 定义marker的位置
        # marker_position = np.array([0.5, 0.0, 0.5])  # 在x=0.5, y=0.0, z=0.5的位置

        # # 添加一个球形marker
        # self.viewer["marker"].set_object(meshcat.geometry.Sphere(0.05), meshcat.geometry.MeshLambertMaterial(color=0xff0000))
        # self.viewer["marker"].set_transform(meshcat.transformations.translation_matrix(marker_position))
        
        
        for point_name, point_position in point_positions.items():
            marker_name = f"marker_{point_name}"
            # placement = pin.SE3(np.eye(3), point_position)
            
            if cube_size:
                self.viewer[marker_name].set_object(meshcat.geometry.Box(cube_size), meshcat.geometry.MeshLambertMaterial(color=color, opacity=opacity))
            elif sphere_radius:
                self.viewer[marker_name].set_object(meshcat.geometry.Sphere(sphere_radius), meshcat.geometry.MeshLambertMaterial(color=color, opacity=opacity))
            self.viewer[marker_name].set_transform(meshcat.transformations.translation_matrix(point_position))
            # print(f"Marked {point_name} at {point_position}")
    
    def ik(
        self,
        left_wrist_mat,
        right_wrist_mat,
    ):
        left_wrist_pose = pin.SE3(left_wrist_mat)
        right_wrist_pose = pin.SE3(right_wrist_mat)

        # Update task targets
        self.right_wrist_task.set_target(right_wrist_pose)
        self.left_wrist_task.set_target(left_wrist_pose)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(self.configuration, self.tasks, self.dt, solver=self.solver, damping=self.damp, safety_break=False)
        self.configuration.integrate_inplace(velocity, self.dt)

        return self.configuration.q, False, 0.0  # left, right

    def solve(self, left_wrist_pose, right_wrist_pose, orientation_cost=None):
        if orientation_cost:
            self.right_wrist_task.set_orientation_cost(orientation_cost)
            self.left_wrist_task.set_orientation_cost(orientation_cost)
            
        left_wrist = left_wrist_pose[:3]
        right_wrist = right_wrist_pose[:3]

        left_wrist_quat = left_wrist_pose[3:][[3, 0, 1, 2]]
        right_wrist_quat = right_wrist_pose[3:][[3, 0, 1, 2]]

        # need wxyz quat
        left_wrist2world_mat = transform_from_pq(
            np.concatenate([left_wrist, left_wrist_quat])
        )
        right_wrist2world_mat = transform_from_pq(
            np.concatenate([right_wrist, right_wrist_quat])
        )

        qpos, success, err = self.ik(left_wrist2world_mat, right_wrist2world_mat)

        if self.fake_wrist_dof:
            return qpos[2:7], qpos[21:26], err
        else:
            return qpos[2:7], qpos[19:24], err


if __name__ == "__main__":
    solver = PinFk()
    joint_arms = [
        "left_shoulder_pitch_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_hand_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_hand_joint",
    ]
    result0 = solver.fk(np.ones_like(np.zeros(36)), joint_arms)
    print(result0["right_hand_joint"])
    result1 = solver.fk(np.zeros_like(np.ones(36)), joint_arms)
    print(result0["right_hand_joint"])
    print(result1["right_hand_joint"])
       