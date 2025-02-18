import pinocchio as pin
import numpy as np
import time

import qpsolvers
from pytransform3d.transformations import (
    transform_from_pq,
    pq_from_transform,
)

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask


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


class ArmPinkIK:
    def __init__(
        self,
        fake_wrist_dof: bool = True,
        eps: float = 0.2,
        MAX_TIME: float = 0.01,
        DT: float = 0.1,
        damp: float = 1e-6,
        in_lp_alpha: float = 0.9,
        orientation_cost: float = 0.5,
    ):
        self.fake_wrist_dof = fake_wrist_dof
        if fake_wrist_dof:
            raise DeprecationWarning("fake_wrist_dof is deprecated. Using normal mode now.")
        else:
            wbc_robot = pin.RobotWrapper.BuildFromURDF(
                filename="./h1_assets/urdf/h1_wrist_pink_v819.urdf",
                package_dirs=["./h1_assets/urdf"],
                root_joint=None,
            )

        locked_joint_id = [
            int(wbc_robot.model.getJointId(joint_name))
            for joint_name in [
                "right_ankle_joint",
                "torso_joint",
                "left_ankle_joint",
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

        # Task target specifications
        posture_task.set_target_from_configuration(self.configuration)

        self.left_wrist_filter = LPConstrainedSE3Filter(in_lp_alpha)
        self.right_wrist_filter = LPConstrainedSE3Filter(in_lp_alpha)

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


        # print("Start_solving!")
        qpos, success, err = self.ik(left_wrist2world_mat, right_wrist2world_mat)

        # TODO: filer after solving?

        if self.fake_wrist_dof:
            return qpos[2:7], qpos[21:26], err
        else:
            return qpos[2:7], qpos[19:24], err
