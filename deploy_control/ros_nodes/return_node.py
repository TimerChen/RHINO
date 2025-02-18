import os
import sys
import time
import subprocess

import numpy as np

sys.path.append(os.curdir)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import (
    LowCmd,
    LowState,
    MotorCmd,
    MotorCmds,
    MotorState,
    MotorStates,
)
from unitree_sdk2py.utils.crc import CRC

from ros_nodes.control_node import WristsController, HeadController

from motion_utils.consts import *

class ReturnNode(Node):
    def __init__(self):
        super().__init__("return_node")

        weight = 0.0
        weight_rate = 0.2

        control_dt = 0.02
        max_joint_velocity = 0.12
        self.terminate = False

        self.msg = LowCmd()  # unitree_go_msg_dds__LowCmd_()
        # self.lowstate = LowState() # unitree_go_msg_dds__LowState_()
        self.fingers_cmd_msg = MotorCmds()
        for _ in range(12):
            self.fingers_cmd_msg.cmds.append(MotorCmd())
        self.lowstate = None
        self.fingers_state = None
        self.crc = CRC()

        # self.final_pos = np.array([0.0, 0.0, 0.0, kPi_2, 0.0, 0.0, 0.0, kPi_2])
        self.wrists_controller = WristsController("/dev/ttyACM0")
        self.head_controller = HeadController("/dev/ttyUSB0")

        """
        http://urdf.robotsfan.com/
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        """

        # arm finalize position(s)
        self.final_pos_traj = np.array([
            [0.0, 0.32, 0.0, -0.64, 0.0, -0.32, 0.0, -0.64],
            [0.0, 0.32, kPi_2, -0.64, 0.0, -0.32, -kPi_2, -0.64],
            [0.0, 0.32, kPi_2, kPi_2, 0.0, -0.32, -kPi_2, kPi_2],
            # [0.0, 0.0, 0.0, kPi_2, 0.0, 0.0, 0.0, kPi_2],
            [0.0, 0.0, -1.3, kPi_2, 0.0, 0.0, 1.3, kPi_2]
            ])
        # self.check_pos = np.array([0.0, 0.0, -1.3, kPi_2, 0.0, 0.0, 1.3, kPi_2])

        self.control_dt = control_dt
        self.max_joint_velocity = max_joint_velocity
        self.weight_rate = weight_rate
        self.max_joint_delta = max_joint_velocity * control_dt
        self.delta_weight = weight_rate * control_dt

        os.system("find /dev -name 'ttyUSB*' -o -name 'ttyACM*' | xargs -I {} chmod 666 {}")
        # bashCommand = "bash /Share/scripts/control_hand.sh"
        
        bashCommand = "/Code/h1_inspire_service/build/inspire_hand -s /dev/ttyUSB1"
        # self.fingers_control_process = subprocess.Popen(bashCommand.split(),stdout=subprocess.PIPE)

        # arm messages
        self.arm_sdk_pub = self.create_publisher(LowCmd, "arm_sdk", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "lowstate", self.lowstate_callback, 10
        )
        # print(self.lowstate_sub, "Subscribed to lowstate")
        # finger messages
        self.fingers_cmd_pub = self.create_publisher(MotorCmds, "/inspire/cmd", 10)
        # The msg received from the inspire hand
        self.fingers_state_sub = self.create_subscription(
            MotorStates, "/inspire/state", self.fingers_state_callback, 10
        )


        self.finialize_arms_part1()
        self.finialize_head()
        self.finialize_wrists()
        self.finialize_fingers()
        self.finialize_arms_part2()
    
    def finialize_arms_part1(self):
        self.set_weight(1.0)
        for final_pos in self.final_pos_traj[:2]:
            self.move_arm_to(final_pos)
            time.sleep(0.5)

    def finialize_arms_part2(self):
        self.set_weight(1.0)
        for final_pos in self.final_pos_traj[2:]:
            self.move_arm_to(final_pos)
            time.sleep(0.5)
        self.set_weight(0.0)
        self.send_msg()

        # self.finalize_robot()

    def finialize_fingers(self):
        self.move_fingers_to_qpos([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        time.sleep(1)
        self.move_fingers_to_qpos([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        time.sleep(1)



    def finialize_wrists(self):
        self.wrists_controller.init_motors(init_pos=[0.0, -0.0])
        [self.wrists_controller.motor_ctrl.disable(m) for m in self.wrists_controller.motors]

    def finialize_head(self):
        self.head_controller.set_position_from_head_yp(np.array([0.3, 0.0]), old_mode=True) # TODO: check the joint angle.
        time.sleep(1.0)
        self.head_controller.disable()

    # ================== FINGER CONTROL ==================

    def move_fingers_to_qpos(self, right_angles, left_angles):
        # print("right", right_angles, left_angles)
        self.get_logger().info(f"Move fingers to qpos: {right_angles}, {left_angles}")
        for i in range(6):
            self.fingers_cmd_msg.cmds[i].q = float(right_angles[i])

        for i in range(6):
            self.fingers_cmd_msg.cmds[i + 6].q = float(left_angles[i])

        self.fingers_cmd_pub.publish(self.fingers_cmd_msg)

    # ================== ARM CONTROL ==================

    def lowstate_callback(self, msg):

        self.lowstate = msg
        # self.get_logger().info('lowstate heard: "%s"' % self.get_arm_joint())


    def fingers_state_callback(self, msg):
        self.fingers_state = msg

    def move_arm_to(self, target_pos=None):
        if target_pos is None:
            target_pos = self.init_pos

        current_jpos_des = self.get_arm_joint()

        for i in range(3000):

            if np.max(np.abs(current_jpos_des - target_pos)) < 0.005:
                break

            for j in range(len(target_pos)):
                current_jpos_des[j] += np.clip(
                    target_pos[j] - current_jpos_des[j],
                    -self.max_joint_delta,
                    self.max_joint_delta,
                )

            self.step(current_jpos_des)
            rclpy.spin_once(self)
            time.sleep(0.002)

    def get_arm_joint(self):
        motor_state = self.get_state()
        return np.array([motor_state[ArmJoints[j]].q for j in range(len(ArmJoints))])

    def get_state(self):
        # msg = self.sub.Read()
        while True:
            if self.lowstate is not None:
                msg = self.lowstate
                return msg.motor_state  # TODO: check if msg is in the right format
            else:
                rclpy.spin_once(self)

    def set_weight(self, weight):
        self.msg.motor_cmd[JointIndex["kNotUsedJoint"]].q = weight

    def step(
        self, action, dq=0.0, kp=60.0, kd=1.5, tau_ff=0.0
    ):  # action = target_joints
        # set control joints
        for j in range(len(action)):
            self.msg.motor_cmd[ArmJoints[j]].q = action[j]
            self.msg.motor_cmd[ArmJoints[j]].dq = dq
            self.msg.motor_cmd[ArmJoints[j]].kp = kp
            self.msg.motor_cmd[ArmJoints[j]].kd = kd
            self.msg.motor_cmd[ArmJoints[j]].tau = tau_ff

        self.send_msg()

    def send_msg(self):
        # self.msg.crc = self.crc.Crc(self.msg)
        if not self.terminate:
            # self.pub.Write(self.msg)
            # print('I am going to publish msg', [self.msg.motor_cmd[ArmJoints[j]].q for j in range(len(ArmJoints))])
            self.arm_sdk_pub.publish(self.msg)
            pass
        else:
            print("ERROR: Robot is terminated")

    # def finalize_robot(self):
        # self.msg.motor_cmd[JointIndex["kNotUsedJoint"]].q = 0.0
        # self.send_msg()
    def destroy_node(self):
        # self.wrists_controller.set_fixed_target([kPi_2, -kPi_2])
        # terminate hand control process
        # self.fingers_control_process.terminate()
        # self.fingers_control_process.wait()
        super().destroy_node()

def main(args=None):
    rclpy.init()
    # try:
    #     node = ControlNode()
    #     rclpy.spin(node)
    #     node.destroy_node()
    # except KeyboardInterrupt:
    #     # node.move_arm_to_init()
    #     print("Keyboard Interrupt")
    #     node.destroy_node()
    #     # sys.exit(0)
    node = ReturnNode()
    # rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
