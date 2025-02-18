from collections import deque
import time
import numpy as np

from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension, ByteMultiArray

class ActionBuffer:
    """
    Recive multiple actions and play them in a fixed fps.
    """

    def __init__(self, init_pos=None, fps=15, bit_size=8) -> None:
        self.fps = fps
        self.frameperiod = 1.0 / fps
        self.buffer = None
        if init_pos is None:
            self.target_pos = np.zeros(bit_size)
        else:
            self.target_pos = init_pos

        self.clear()
        
    def clear(self):
        self.reset()
        self.buffer = deque()

    def reset(self):
        self.nextframe = time.time() + self.frameperiod

    def put(self, pos, fps=None):
        if self.target_pos is None:
            self.target_pos = pos
        while time.time() > self.nextframe:
            self._fresh_pos()
            self.nextframe += self.frameperiod
        if fps is not None:
            self.buffer.append((pos, fps))

    def _fresh_pos(self):
        if len(self.buffer) > 0:
            target_pos = self.buffer.popleft()
            if isinstance(target_pos, tuple):
                frameperiod = 1. / target_pos[-1]
                target_pos = target_pos[0]
            else:
                frameperiod = self.frameperiod
                
            self.target_pos = target_pos
            return frameperiod
        return self.frameperiod

    def get_current_target(self):
        while time.time() > self.nextframe:
            frameperiod = self._fresh_pos()
            self.nextframe += frameperiod
        # print("########## queue size:", self.buffer.qsize())
        return self.target_pos

def pack_timestamp_data(data, timestamp, data_dtype=np.float32, timestamp_dtype=np.int32):
    """
    Pack the data with timestamp into msg.
    """
    # timestamp is int32, data is float32
    data_bytes = data.tobytes()
    timestamp_bytes = timestamp.tobytes()
    msg = ByteMultiArray()
    msg.data = timestamp_bytes + data_bytes
    msg.layout = MultiArrayLayout()
    msg.layout.dim = [MultiArrayDimension()]
    msg.layout.dim[0].size = len(data)
    msg.layout.dim[0].stride = len(data)
    msg.layout.dim[0].label = 'timestamp+data'
    return msg

def unpack_timestamp_data(msg, data_dtype=np.float32, timestamp_dtype=np.int32):
    """
    Extract the timestamp from the data.
    """
    raw_data = msg.data
    ts_bytes_size = np.dtype(timestamp_dtype).itemsize
    timestamp = np.frombuffer(raw_data[:ts_bytes_size], dtype=timestamp_dtype)
    data = np.frombuffer(raw_data[ts_bytes_size:], dtype=data_dtype)
    return timestamp, data