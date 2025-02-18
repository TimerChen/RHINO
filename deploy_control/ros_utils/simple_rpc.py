import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

class SimpleRPC():
    """
        call process:
        caller:
            - rpc.call(no_wait=True)
            - <do others>
            - rpc.wait()
            - <get result>
        runner:
            - rpc.wait() or rpc.check()
            - <do others>
            - rpc.return_called()
    
    """
    def __init__(self, buffer):
        self.call_event = mp.Event()
        self.return_event = mp.Event()
        self.quit_event = mp.Event()
        self.buffer = buffer
        self.is_caller = True
        self.shm_buffer = []
        
    def set_caller(self, is_caller):
        for k,v in self.buffer.items():
            if isinstance(v, tuple):
                shm_name, shm_shape = v
                shm = shared_memory.SharedMemory(name=shm_name,)
                self.shm_buffer.append(shm)
                np_data = np.ndarray(shm_shape, dtype=np.uint8, buffer=shm.buf)
                self.buffer[k] = np_data
        self.is_caller = is_caller
        
    def call(self, no_wait=False):
        self.call_event.set()
        if not no_wait:
            self.return_event.wait()
            self.return_event.clear()
    
    def wait(self):
        if not self.is_caller:
            self.call_event.wait()
            self.call_event.clear()
        else:
            self.return_event.wait(1)
            if not self.return_event.is_set():
                assert False, f"RPC timeout"
            self.return_event.clear()
        
    def check(self):
        if self.is_caller:
            return self.return_event.is_set()
        return self.call_event.is_set()
    
    def response(self):
        self.return_event.set()
        
    def is_quit(self):
        return self.quit_event.is_set()
        
    def __del__(self):
        if self.is_caller:
            self.quit_event.set()
            self.call_event.set()