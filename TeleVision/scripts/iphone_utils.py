import multiprocessing as mp
import numpy as np
from multiprocessing import resource_tracker, shared_memory

def get_shm_size(shape, dtype):
   return np.prod(shape) * dtype().itemsize

class IphoneCamSHM:
    def __init__(self, iphone_mode, shm_name="iphone_img", ):
        self.iphone_mode = iphone_mode
        shapes = [(3, 480, 640, ), (480, 640)]
        sizes = [get_shm_size(shapes[0], dtype=np.uint8),
                 get_shm_size(shapes[1], dtype=np.float32)]
        self.shm = shared_memory.SharedMemory(
            name=shm_name, create=False, size=np.sum(sizes)
        )
        # if not create_shm:
        resource_tracker.unregister(f"/{shm_name}", 'shared_memory')

        self.img_shm = np.ndarray(
            shapes[0],
            dtype=np.uint8,
            buffer=self.shm.buf,
            offset=0,
        )
        self.depth_shm = np.ndarray(
            shapes[1],
            dtype=np.float32,
            buffer=self.shm.buf,
            offset=get_shm_size(shapes[0], dtype=np.uint8),
        )

    def get_image(self):
        return self.img_shm.copy(), self.depth_shm.copy()