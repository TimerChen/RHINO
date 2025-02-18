from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)
from dynamixel_sdk_custom_interfaces.msg import SetPosition
from dynamixel_sdk_custom_interfaces.srv import GetPosition
import numpy as np
import time
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
#! rewrite 
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_POSITION = 140 #Position Trajectory(140)
LEN_PRESENT_POSITION = 4

class HeadController:
    def __init__(self, serial_name) -> None:
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        self.PROTOCOL_VERSION = 2.0

        self.serial_name = serial_name
        self.baudrate = 57600

        self._portHandler = PortHandler(self.serial_name)
        self._packetHandler = PacketHandler(2.0)

        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # 1, 3/2
        self.joint_offset = np.array([0.95 * np.pi, 1.6 * np.pi])
        # Set Port
        dxl_comm_result = self._portHandler.openPort()
        self.check_and_log(dxl_comm_result, "open the port", exit_when_fail=True)

        dxl_comm_result = self._portHandler.setBaudRate(self.baudrate)
        self.check_and_log(dxl_comm_result, "set the baudrate", exit_when_fail=True)

        for id in [1,2]:
            dxl_addparam_result = self._groupSyncRead.addParam(id)
            self.check_and_log(dxl_addparam_result, f"addParam for Dynamixel with ID {id}")
        
        self.enable()

    def enable(self):
        # Set Dynamixel
        for id in [1, 2]:
            dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                self._portHandler,
                id,
                self.ADDR_OPERATING_MODE,
                3,
            )
            self.check_and_log(
                dxl_comm_result == COMM_SUCCESS, "set Position Control Mode"
            )

            dxl_comm_result, dxl_erorr = self._packetHandler.write1ByteTxRx(
                self._portHandler, id, self.ADDR_TORQUE_ENABLE, 1
            )
            self.check_and_log(dxl_comm_result == COMM_SUCCESS, "enable torque")

    
    def disable(self):
        for id in [1, 2]:
            dxl_comm_result, dxl_erorr = self._packetHandler.write1ByteTxRx(
                self._portHandler, id, self.ADDR_TORQUE_ENABLE, 0
            )
            self.check_and_log(dxl_comm_result == COMM_SUCCESS, "disable torque")

    def check_and_log(self, succ_condition, action_info, exit_when_fail=False):
        if succ_condition:
            print(f"Succeeded to {action_info}.")
        else:
            print(f"Failed to {action_info}.")
            if exit_when_fail:
                exit()

    def set_position_from_head_yp(self, head_yp: np.ndarray, 
                                  human_offset=False, old_mode=False,):
        """ only for human control, we give an offset to the joint angle """
        # NOTE: in old mode, the offset is add in control node
        if old_mode:
            set_value = (head_yp + self.joint_offset).tolist()
            set_value = np.mod(set_value, 2 * np.pi)
            if human_offset:
                # set_value[1] = np.clip(set_value[1], 4.47+1.1, 5.30+0.4)
                set_value[1] = np.clip(set_value[1], 4.47+1.1, 4.47+1.1)
        else:
            set_value = head_yp.tolist()
        now = time.time()
        for idx, ang in enumerate(set_value):
            position_value = int(ang * 2048 / np.pi)
            # print("[set_position_from_head_yp]",idx+1, position_value, ang)
            msg = SetPosition(id=idx + 1, position=position_value)
            # now = time.time()
            self.set_position(msg)
        dxl_comm_result = self._groupSyncWrite.txPacket()
        # self.check_and_log(dxl_comm_result == COMM_SUCCESS, "syncwrite goal position")
        self._groupSyncWrite.clearParam()
        # print("[set_position_from_head_yp]", time.time()-now)

    def set_position(self, msg: SetPosition):
        dxl_id = msg.id
        goal_position = msg.position

        param_goal_position = [
            DXL_LOBYTE(DXL_LOWORD(goal_position)),
            DXL_HIBYTE(DXL_LOWORD(goal_position)),
            DXL_LOBYTE(DXL_HIWORD(goal_position)),
            DXL_HIBYTE(DXL_HIWORD(goal_position)),
        ]
        # print(goal_position)
        dxl_addparam_result = self._groupSyncWrite.addParam(
            dxl_id, param_goal_position
        )

        # self.check_and_log(dxl_addparam_result, f"set joint angle for Dynamixel with ID {dxl_id}")
        return

        dxl_comm_result, dxl_error = self._packetHandler.write4ByteTxRx(
            self._portHandler,
            msg.id,
            self.ADDR_GOAL_POSITION,
            goal_position,
        )

        if dxl_comm_result != COMM_SUCCESS:
            print(self._packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(self._packetHandler.getRxPacketError(dxl_error))
        else:
            pass
            # print(f"Set [ID: {msg.id}] [GOAL POSITION: {msg.position}]")
            # print("[set_position_from_head_yp]", time.time()-now)

    def get_position_as_head_yp(self):
        head_yp = [0.0, 0.0] # np.zeros(2)
        dxl_comm_result = self._groupSyncRead.txRxPacket()
        # self.check_and_log(dxl_comm_result == COMM_SUCCESS, "syncread present position")

        for idx in [1,2]:
            position_value = self.get_position(idx)
            ang = position_value * np.pi / 2048 # TODO: check the range of joint angle.
            # head_yp[idx-1] = ang - self.joint_offset[idx-1]
            head_yp[idx-1] = ang
        return head_yp
    
    def get_position(self, request_id):
        if self._groupSyncRead.isAvailable(
            request_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        ):
            angle = self._groupSyncRead.getData(
                request_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            angle = np.int32(np.uint32(angle))
            # print(f"Get [ID: {request_id}] [Present_Position: {angle}]") # example: 2155,3632
            return angle
        else:
            raise RuntimeError(
                f"Failed to get joint angles for Dynamixel with ID {request_id}"
            )

        (
            present_position,
            datadxl_comm_result,
            dxl_error,
        ) = self._packetHandler.read4ByteTxRx(
            self._portHandler,
            request_id,
            self.ADDR_PRESENT_POSITION,
        )
        # print(f"Get [ID: {request_id}] [Present_Position: {present_position}]")
        # respond.position = present_position
        return present_position