import cv2
import numpy as np
from multiprocessing import shared_memory
import time
import subprocess
import re

class CameraFromVideo:
    def __init__(self, video_id, camera_scale, fps, init_cap):
        self.cap = init_cap
        self.frame_counter = 0

    def read(self):
        ret, frame = self.cap.read()
        self.frame_counter += 1
        # if not ret:
        #     return False, None
        if self.frame_counter == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            self.frame_counter = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return ret, frame

    def release(self):
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)
    
    def set(self, prop, value):
        self.cap.set(prop, value)

    def wait(self):
        if self.now < self.nextframe:
            time.sleep(self.nextframe - self.now)
            self.now = time.time()
        self.nextframe += self.frameperiod

class ShmCamera():
    def __init__(self, shm_name, shm_shape) -> None:
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.shared_array = np.ndarray(shm_shape, dtype=np.uint8, buffer=self.shm.buf)

    def read(self):
        return True, self.shared_array.copy()
    
def reset_video(cap, start_frame=0):
    # set pointer to the beginning of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

def open_camera(VIDEO_ID, camera_scale=2, fps=15, real_cap=False):
    cap = cv2.VideoCapture(VIDEO_ID)
    if not isinstance(VIDEO_ID, str):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920//camera_scale)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080//camera_scale)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560//camera_scale)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440//camera_scale)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        if real_cap:
            return cap
        cap = CameraFromVideo(VIDEO_ID, camera_scale, fps, cap)

    return cap

def test_open_camera(VIDEO_ID, camera_scale=1, fps=15, resolution=(2560, 1440)):
    # cap1 = open_camera(0, camera_scale, fps)
    cap = open_camera(VIDEO_ID, camera_scale, fps)
    ret, frame = cap.read()
    # ret, frame2 = cap1.read()
    # cv2.imshow(f"test{VIDEO_ID}", frame)
    # cv2.waitKey(100)
    # cv2.imshow(f"test{VIDEO_ID}", frame2)
    # cv2.destroyAllWindows()
    cap_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if ret:
        print("open camera", VIDEO_ID, "resolution", cap_res)
    res_same = [cap_res[i] == resolution[i] for i in range(2)]
    res_same = all(res_same)
    cap.release()
    return ret, res_same, frame
    # cap1.release()

def detect_available_camera():
    # see from /dev/vedio*
    cmd = "ls /dev/video*"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # output = subprocess.check_output(["ls", "/dev/video*"], text=True)
    output, error = process.communicate()
    output = output.decode("utf-8").split("\n")
    output = [re.findall(r"\d+", i) for i in output if i]
    output = [int(i[0]) for i in output]
    print("[INFO] Detected cameras:", output)
    vids = []
    for i in output:
        ret, res_same, frame = test_open_camera(i)
        if ret and res_same:
            vids.append(i)

    print(f"[INFO] Available Cameras {vids}")
    return vids

def assign_camera_id(vids):
    ordered_vids = [None, None]
    for i in vids:
        cap = open_camera(i)
        while True:
            ret, frame = cap.read()
            cv2.imshow(f"Press: robo(0), human(1)", frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("0") or key == ord("1"):
                break
        cap.release()
        ordered_vids[int(chr(key))] = i
    # cv2.destroyAllWindows()
    cv2.destroyWindow(f"Press: robo(0), human(1)")
    return ordered_vids


def get_shm_size(shape, dtype):
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    return nbytes


def image_producer(shared_memory_name, shape, camera_kwargs):
    
    # see from /dev/vedio*
    cap = open_camera(**camera_kwargs)
    file_camera = isinstance(camera_kwargs["VIDEO_ID"], str)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frameperiod=1.0/fps
    
    now=time.time()
    nextframe=now+frameperiod
        
    shm = shared_memory.SharedMemory(name=shared_memory_name)
    shared_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        frame_counter += 1
        if not ret:
            break
        shared_array[:] = frame  # 复制图像到共享内存
        
        # open it again if it is closed
        if file_camera and frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if now<nextframe:
            time.sleep(nextframe-now)
            now=time.time()
        nextframe += frameperiod
            
    cap.release()
    shm.close()


def image_display(processed_memory_name, shape, frame_scale=3):
    shm = shared_memory.SharedMemory(name=processed_memory_name)
    processed_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    while True:
        if np.any(processed_array):  # 检查共享内存是否已被写入
            frame = processed_array.copy()
            # print("fame", frame.shape)
            frame = cv2.resize(frame, (frame.shape[1]//frame_scale, 
                                       frame.shape[0]//frame_scale))
            cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    shm.close()
    cv2.destroyAllWindows()