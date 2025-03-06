import os
import argparse
import cv2
import sys


sys.path.append(os.curdir)
sys.path.append(os.path.join(os.curdir, "ros_nodes"))

import numpy as np
from PIL import Image
import subprocess
from scipy.spatial.transform import Rotation as R
import torch
import h5py
import pickle as pkl

from zed_module.zed_module import ZedModule
# from deploy_control.scripts.process_demo_video_hand import HandDetector
from ros_utils.display_utils import get_shm_size
from zed_module.yolo_detector import draw_hand_obj

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def axis_angle_to_cont6d(axis_angle):
    rotation = R.from_rotvec(axis_angle)
    rotation_mat = rotation.as_matrix()
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


class ProcessDataZed:
    def __init__(
        self,
        overwrite=False,
        human=True,
        humanoid=True,
        zed_img_shm_info=None,
        hand_details_only=False,
        vis_hand=False,
        vis_body=False,
        scenario="dining",
        one_data=False,
    ):
        self.overwrite = overwrite
        self.dir = None
        self.humanoid = True
        self.process_human = human
        self.process_humanoid = humanoid
        self._zed_module = None
        self.zed_img_shm_info = zed_img_shm_info
        self.hand_details_only = hand_details_only
        self.scenario = scenario
        self.sk_image_shape = [640, 640]
        self.upper_body_joints = [16, 17, 18, 19, 20, 21, 22, 23]
        self.vis_body = vis_body
        self.one_data = one_data

    def process_task(self, task_path):
        for root, dirs, _ in os.walk(task_path):
            dirs.sort(key=lambda x: int(x.split("-")[-1]))
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                self.process_dir(dir_path)
                if self.one_data:
                    print("[INFO] Usi·ng one_data mode, break.")
                    break
            break

    def process_dir(self, dir_path):
        self.dir = dir_path
        rgb_path = os.path.join(self.dir, "zed_left.mp4")

        # if self.process_human:
        print("process dir", dir_path)
        if os.path.exists(rgb_path) and not self.overwrite:
            return
        
        # if not os.path.exists(data_path2) or self.overwrite:
        print(f"Processing human {dir_path}")
        self.humanoid = False
        self.process_video_zed()
                
    def get_zed_module(self, svo_file):
        """
        no obj detection, use low detect level for fast
        """
        detect_level="low" # 38 nodes
        model_level="medium"
        if self._zed_module is None:
            self._zed_module =ZedModule(svo_file=svo_file, visualize=True, 
                            model_level=model_level, detect_level=detect_level,
                            img_shm_info=self.zed_img_shm_info, obj_detection=False,
                            scenario=self.scenario)
        else:
            self._zed_module.reload(svo_file)
        return self._zed_module, detect_level, model_level
    
    def process_video_zed(self, confidence_threshold=0.5):
        # self.align_the_ts()
        # return
        
        svo_file = os.path.join(self.dir, "zed.svo2")
        zed_module, detect_level, model_level = self.get_zed_module(svo_file)
        # self.hand_detector.zed_module = zed_module
        # self.hand_detector.reset_track_info()
        matched_frames = self.match_frames()
        # if self.hand_detector is not None:
        #     self.hand_detector.zed_module = zed_module
            
        bodies = 0
        err_cnt = 0
        rots = np.zeros((6, 4))
        rots[:] = np.nan
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if not self.hand_details_only:
            matched_zed_out = cv2.VideoWriter(os.path.join(self.dir, "zed_left.mp4"), fourcc, 30, (1280, 720))
        
        print("matched_frames", len(matched_frames))
        
        parts = ["body", "hand", "hand_details"]
        if self.hand_details_only:
            parts = parts[-1:]
        
        fid = -1
        m_fid = 0
        while True:
            if fid % 100 == 0:
                print("processing frame", fid)
            fid += 1
            
            imgs = zed_module.get_image(render=False, skip_body=True)

            if imgs[0] is None:
                break
            img = imgs[0].get_data()
            if matched_frames[m_fid] != fid:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # all_left_imgs.append(img)
            
            while m_fid < len(matched_frames) and matched_frames[m_fid] == fid:
                if not self.hand_details_only:
                    matched_zed_out.write(img)
                
                m_fid += 1
                
        while m_fid < len(matched_frames):
            if not self.hand_details_only:
                matched_zed_out.write(img)
                
            m_fid += 1
            
        # assert len(hand_details_imgs) == len(matched_frames), f"{len(hand_details_imgs)} != {len(matched_frames)}"
        
        if not self.hand_details_only:
            matched_zed_out.release()

    def match_frames(self, pad_missing=True):
        # align with the video
        input_file = os.path.join(self.dir, "data.hdf5")
        file = h5py.File(input_file, 'r')
        zed_start_timestamp = np.array(file["video_start_stamp"], dtype=np.int64).item()
        if not os.path.exists(os.path.join(self.dir, "tag_count_log.pkl")):
            with h5py.File(os.path.join(self.dir, "data.hdf5"), "r") as f:
                timesteps = np.array(f["timestamp"])
            return np.arange(len(timesteps))
        
        if not os.path.exists(os.path.join(self.dir, "tag_count_log.pkl")):
            data_len = len(file["timestamp"])
            print(f"[INFO] No tag_count_log.pkl, return all frames: {data_len}")
            return list(range(data_len))
        
        # react timestamp list
        with open(os.path.join(self.dir, "tag_count_log.pkl"), "rb") as f:
            tag_count_log = pkl.load(f)
            react_timestamps = tag_count_log["timestamp"]

        matched_frame_id = []
        match_cnt = 0
        max_len = int(1e8)
        for tt in react_timestamps:
            t = tt - zed_start_timestamp
            if t < 0 or t >= max_len:
                if pad_missing:
                    t_ = np.clip(t, 0, max_len - 1)
                    # matched_rots.append(all_rots[t_])
                    # matched_images.append(all_left_imgs[t_])
                    matched_frame_id.append(t_)
                else:
                    # matched_rots.append(nan_rot)
                    # matched_images.append(None)
                    assert False, f"not support no padding"
            else:
                # matched_rots.append(all_rots[t])
                # matched_images.append(all_left_imgs[t])
                matched_frame_id.append(t)
                match_cnt += 1
        # print(f"[INFO] matched_rots: {match_cnt}: {match_cnt / len(react_timestamps)}")
        # matched_rots = np.stack(matched_rots, axis=0)
        matched_frame_id = np.stack(matched_frame_id, axis=0)

        return matched_frame_id
        # save the data

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description="Process some videos.")
    parser.add_argument("--root", type=str, help="The root directory of the task")
    parser.add_argument("--task", type=str, help="The task to process")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    # parser.add_argument("--human", action="store_true", help="Process human data")
    # parser.add_argument("--humanoid", action="store_true", help="Process humanoid data")
    parser.add_argument("--ep", type=str, help="The epoch to process")
    parser.add_argument("--loop", action="store_true", help="Detect the task")
    parser.add_argument("--hand_details_only", action="store_true", help="Process hand details only")
    parser.add_argument("--vis_hand", action="store_true", help="Visualize the results")
    parser.add_argument("--vis_body", action="store_true", help="Visualize the results")
    parser.add_argument("--scenario", type=str, default="dining", choices=["dining", "office", "0", "1"],
                        help="react scenario, 0: dining, 1: office")
    parser.add_argument("--one_data", action="store_true", help="Process one data per task")
    args = parser.parse_args()
    
    zed_img_shm_info = ZedModule.create_zed_img_shm("zed_img_process")
    zed_img_shm_info, zed_img_shm = zed_img_shm_info[:2], zed_img_shm_info[2]
    
    processer = ProcessDataZed(overwrite=args.overwrite, human=True, humanoid=False,
                               zed_img_shm_info=zed_img_shm_info,
                               hand_details_only=args.hand_details_only,
                               vis_hand=args.vis_hand,
                               vis_body=args.vis_body,
                               scenario=args.scenario,
                               one_data=args.one_data)
    
    if args.loop:
        while True:
            command = [
                'rsync',
                '-avp',
                # '--exclude', 'cheers',
                # '--exclude', 'handshake',
                # '--exclude', 'thumbup',
                # 'robo2:/home/apex/Dataset/react_data/',
                'robo2:/home/apex/Code/webcam2motion/react_data_new_robo2/',
                '/home/jxchen/Code/webcam2motion/react_data_new'
            ]
            subprocess.run(command)
            for dir_name in os.listdir(args.root):
                task_dir = os.path.join(args.root, dir_name)
                if os.path.isdir(task_dir):
                    for sub_dir_name in os.listdir(task_dir):
                        dir = os.path.join(task_dir, sub_dir_name)
                        processer.process_dir(dir)
    elif args.task and not args.ep:
        # 假设你的任务目录结构是 /home/zbzhu/Code/webcam2motion/motion_data/<task_name>
        task_path = os.path.join(args.root, args.task)
        
        print('task path', task_path, args.root, args.task)
        processer.process_task(task_path)
    elif args.ep:
        task_path = os.path.join(args.root, args.task)
        dir_path = os.path.join(task_path, args.ep)
        processer.process_dir(dir_path)
    else:
        for dir_name in os.listdir(args.root):
            task_dir = os.path.join(args.root, dir_name)
            print("Processing", task_dir)
            processer.process_task(task_dir)
            # if os.path.isdir(task_dir):
            #     for sub_dir_name in os.listdir(task_dir):
            #         dir = os.path.join(task_dir, sub_dir_name)
            #         processer.process_dir(dir)
    # del processer._zed_module.detect_rpc
    zed_img_shm.close()
    zed_img_shm.unlink()

