from motion_utils.consts import *
from ros_nodes.wrists_motor_controller import MotorControl, Motor, DM_Motor_Type

from serial import Serial
import time
import numpy as np

class WristsController:
    def __init__(self, serial_name) -> None:
        self.motors = []
        self.serial_name = serial_name

    def init_motors(self, init_pos=[kPi_2, -kPi_2], no_control=False):
        self.motors = [
            Motor(DM_Motor_Type.DM4310, 0x01, 0x11),  # left wrist, q~[0,1]
            Motor(DM_Motor_Type.DM4310, 0x02, 0x12),  # right wrist, q~[-1,0]
        ]
        self.serial_device = Serial(self.serial_name, 921600, timeout=0.5)
        # serial_device = FakeSerial()

        self.motor_ctrl = MotorControl(serial_device=self.serial_device, debug=False)
        [self.motor_ctrl.addMotor(m) for m in self.motors]
        [self.motor_ctrl.disable(m) for m in self.motors]
        # [ self.motor_ctrl.zero_position(m) for m in self.motors]
        if no_control:
            return
        
        [self.motor_ctrl.enable(m) for m in self.motors]
        print("Ready for control...")
        self.set_fixed_target(init_pos)

    def set_fixed_target(self, target_qs): 
        for i in range(200):
            if (
                np.max(
                    [
                        np.abs(self.motors[0].getPosition() - target_qs[0]),
                        np.abs(self.motors[1].getPosition() - target_qs[1]),
                    ]
                )
                < 0.1
            ):
                # print("[set_fixed_target]", target_qs, "reached")
                break
            # print("[set_fixed_target]", self.motors[0].getPosition(), np.abs(self.motors[1].getPosition()))
            self.ctrl(target_qs)
            time.sleep(0.001)

    def test_action(self):
        qlist = [-0.3, 0, 0.3, 0]
        for q in qlist:
            print("execute MIT control", q)

            [self.motor_ctrl.controlMIT(m, 5, 0.4, q, 0, 0.1) for m in self.motors]
            time.sleep(1)

    def ctrl(self, qs):
        """
        qs: [left_wrist, right_wrist], q~[0, 5.75]
        """
        qs = qs.copy()
        qs[0] = np.clip(qs[0], 0.1, 4.5)
        qs[1] = np.clip(qs[1], -4.5, -0.1)

        state = [0, 0]
        state[0] = self.motors[0].getPosition()
        state[1] = self.motors[1].getPosition()
        wrist_delta = 0.5
        qs[0] = np.clip(qs[0], state[0] - wrist_delta, state[0] + wrist_delta)
        qs[1] = np.clip(qs[1], state[1] - wrist_delta, state[1] + wrist_delta)
        self.motor_ctrl.controlMIT(self.motors[0], 5, 0.4, qs[0], 0, 0.1)
        self.motor_ctrl.controlMIT(self.motors[1], 5, 0.4, qs[1], 0, 0.1)
        # print("[ctrl]", self.motors[0].getPosition(), np.abs(self.motors[1].getPosition()))

    def get_state(self):
        info = {
            "q": [m.getPosition() for m in self.motors],
            "dq": [m.getVelocity() for m in self.motors],
            "torque": [m.getTorque() for m in self.motors],
        }
        return info

    # def __del__(self):
    #     [self.motor_ctrl.disable(m) for m in self.motors]
    #     self.serial_device.close()