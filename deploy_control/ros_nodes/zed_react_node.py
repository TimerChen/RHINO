import os
import sys
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, ByteMultiArray

import multiprocessing as mp
from multiprocessing import shared_memory

import cv2
import time
from datetime import datetime
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
# from urdfpy import URDF

sys.path.append(os.curdir)
sys.path.append(os.path.join(os.curdir, "ros_nodes"))
# from ros_nodes.zed_node import ZedModule
from zed_module.zed_module import ZedModule
from motion_utils.hand_detector import HandDetector
from ros_utils.display_utils import get_shm_size
from ros_utils.fps_node import FpsNode


QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
    
def buffer_processer(node_kwargs, 
                     fps=30, max_buffer=35, put_every=1):
    """stack [max_buffer] frames to reaction model"""
    
    # shm = shared_memory.SharedMemory(name=share_memory_name)
    # processed_array = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    rclpy.init()
    node = ReactZedNode(fps, **node_kwargs)
    # while True:
    #     node.time_callback()
    rclpy.spin(node)


class ReactZedNode(FpsNode):
    def __init__(self, fps,  
                 topic_name="fingers_qpos", 
                 img_shm_info=None,
                 body_shm_info=None,
                 svo_file=None,
                 visualize=False,
                 enable_hand_details=True,
                 scenario="dining",
                 reset_event=None):
        super().__init__("react_zed_node")
        self.reset_event = reset_event
        self.publisher_old = self.create_publisher(Float32MultiArray, "human_pose/c1", QOS_PROFILE)
        self.publisher_ = self.create_publisher(ByteMultiArray, topic_name, QOS_PROFILE)
        self.hand_position_pub = self.create_publisher(Float32MultiArray, "hand_position", QOS_PROFILE)
        self.hand_position_msg = Float32MultiArray()
        self.zed_pose_pub = self.create_publisher(Float32MultiArray, "zed_pose", QOS_PROFILE)
        self.zed_pose_msg = Float32MultiArray()
        self.hand_position_msg.data = [0.]*6
        self.h1_head_qpos = np.array([+0.95*np.pi, 5.44])
        self.h1_state_subscriber = self.create_subscription(Float32MultiArray, "all_states", self.h1_state_callback, QOS_PROFILE)
        
        self.target_pos = None
        self.hand_details = None
        self.scenario = scenario
        # self.np_shm = np_shm
        self.enable_hand_details = enable_hand_details
        self.init_zed_processor(img_shm_info, body_shm_info, svo_file, visualize)
        self.timer = self.create_timer(1./fps, self.time_callback)
        self.reset_timer = None
        
    def h1_state_callback(self, msg):
        self.h1_head_qpos = np.array(msg.data[:2])

    def init_zed_position(self):
        head_pub = self.create_publisher(Float32MultiArray, "all_rt_qpos", QOS_PROFILE)
        head_pub2 = self.create_publisher(Float32MultiArray, "all_qpos", QOS_PROFILE)
        # self.head_msg.data = [1.,1.,1.] + ypr + [0.]*(10 + 24)
        
        head_msg = Float32MultiArray()

        ypr = [np.pi, 5.44]
        # ypr = [0.95*np.pi, 5.44]
        # head_msg.data = [1.,1.,1.] + ypr + [0.]*(10 + 24)
        head_msg.data = [1.,0.,0.] + ypr + [0.]*(10 + 24)
        head_pub.publish(head_msg)
        time.sleep(0.1)
        head_msg = Float32MultiArray()
        head_msg.data = [1.,0.,0.] + ypr + [0.]*(10 + 24) + [0.]*4 + [30.]
        head_pub2.publish(head_msg)
        time.sleep(0.1)
        print("ready to spin", head_msg.data)
        rclpy.spin_once(self)
        time.sleep(0.1)
        
    def reset_zed_position(self):
        
        if self.reset_timer is None:
            self.reset_timer = time.time()
            head_pub = self.create_publisher(Float32MultiArray, "all_rt_qpos", QOS_PROFILE)
            head_pub2 = self.create_publisher(Float32MultiArray, "all_qpos", QOS_PROFILE)
            # self.head_msg.data = [1.,1.,1.] + ypr + [0.]*(10 + 24)
            
            head_msg = Float32MultiArray()

            ypr = [np.pi, 5.44]
            # ypr = [0.95*np.pi, 5.44]
            # head_msg.data = [1.,1.,1.] + ypr + [0.]*(10 + 24)
            head_msg.data = [1.,0.,0.] + ypr + [0.]*(10 + 24)
            head_pub.publish(head_msg)
            head_msg = Float32MultiArray()
            head_msg.data = [1.,0.,0.] + ypr + [0.]*(10 + 24) + [0.]*4 + [30.]
            head_pub2.publish(head_msg)
            time.sleep(0.1)
            print("back head to forward")
        elif time.time() - self.reset_timer > 3:
            self.zed_module.reset_positional_tracking()
            self.reset_timer = None
            self.reset_event.clear()
        
    def init_zed_processor(self, img_shm_info, body_shm_info, svo_file, visualize):
        self.init_zed_position()

        self.detect_level="high"
        self.confidence_threshold = 0.5
        self.visualize = visualize
        self.body_err_cnt = 0
        zed_module = ZedModule(svo_file=svo_file, visualize=visualize, 
                                obj_detection=True,
                                detect_level=self.detect_level, 
                                model_level="medium", auto_replay=True,
                                img_shm_info=img_shm_info,
                                async_mode=False, scenario=self.scenario)
        self.hand_detector = HandDetector(zed_module=zed_module, hand_vis_every=-1)
        body_oppo_name, body_shape = body_shm_info
        # body_shm = shared_memory.SharedMemory(name=body_oppo_name)
        # body_array = np.ndarray(body_shape, dtype=np.float32, buffer=body_shm.buf)
        body_array = np.zeros(body_shape, dtype=np.float32)

        bodies = 0
        err_cnt = 0
        rots = np.zeros((6, 4))
        # rots[:] = np.nan
        rots[:, 0] = 1
        nan_rot =rots.copy()
        sixd_rot = np.zeros((6, 6))
        # all_left_imgs = []

        # image_shm_name, image_shm_shape = img_shm_info
        # zed_image_shm = shared_memory.SharedMemory(name=image_shm_name)
        # zed_image = np.ndarray(image_shm_shape, dtype=np.uint8, buffer=zed_image_shm.buf)

        fid = -1
        put_i = 0
        last_t = []
        self.last_arm_rots = rots
        self.zed_module = zed_module
        self.body_array = body_array
        self.fid = fid
        self.zed_imgs = None

    def time_callback(self):
        self.count("ReactZed Main")
        zed_module = self.zed_module
            
        if self.reset_event.is_set():
            self.reset_zed_position()
            
        rots = self.last_arm_rots
        fid = self.fid
        detect_level = self.detect_level
        confidence_threshold = self.confidence_threshold
        obj_vis = self.visualize

        imgs = zed_module.get_image(render=False, skip_body=True)
        self.zed_imgs = imgs

        if imgs[0] is None:
            return
        fid += 1
        img = imgs[0].get_data()
        zed_module.detect_obj(render=obj_vis, call_mode="async")
        zed_module.detect_body(render=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # # put image into shared memory
        body_info = zed_module.get_joint_info()
        if len(body_info) != 1:
            # err_cnt += 1
            # print(f"[Warning] Number of bodies detected is not 1, but {len(body_info)}")
            pass
        ismiss = True                
        if len(body_info) > 0:
            # track arms
            if detect_level == 'high':
                arm_list = [12, 14, 16, 13, 15, 17]
            elif detect_level == 'medium':
                arm_list = [5, 6, 7, 12, 13, 14]

            for i, body_i in enumerate(arm_list):
                if (body_info[0][2][body_i] >= confidence_threshold and not np.isnan(body_info[0][2][body_i]).any()) or np.isnan(rots[i]).any():
                    rots[i] = body_info[0][1][body_i]
        # track hands
        if fid %6 == 0:
            hand_image, hand_data, hand_raw_data, hand_success, = self.hand_detector.process_zed_current_img(img)
            self.body_array[36:] = hand_data[[0, 1, 4, 6, 8, 10, 12, 13, 16, 18, 20, 22]].flatten() 
            # hand_cam_t = np.clip(hand_cam_t, -3.0, 3.0).astype(np.float32).reshape(-1)
            # self.hand_position_msg.data = hand_cam_t.tolist()
            # self.hand_position_pub.publish(self.hand_position_msg)
            if self.visualize:
                cv2.imshow("hand_image", hand_image)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
        # from quat to 6d
        # ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
        sixd_rot = R.from_quat(rots).as_matrix()[..., :2, :].reshape(-1, 6)  # (6, 6)
        self.body_array[:36] = sixd_rot.flatten()
        
        if self.enable_hand_details:
            zed_module.detect_obj(render=obj_vis, call_mode="get")
            info = zed_module.get_hand_details(img)
            head_info = zed_module.get_head_details()
            cam_rot, cam_trans = zed_module.get_it()
            
            # image
            res = 128
            himg = np.zeros((2, res, res, 3), dtype=np.uint8)
            
            for i in range(2):
                if "hand_imgs" not in info or info["hand_imgs"][i] is None or \
                    info["hand_imgs"][i].shape[0] == 0 or info["hand_imgs"][i].shape[1] == 0:
                    himg[i] = np.zeros((res, res, 3), dtype=np.uint8)
                else:
                    himg[i] = cv2.resize(info["hand_imgs"][i], (res, res))
                
            himg = himg.reshape(2*res, res, 3)
            
            # hand_details = 
            hand_imgs = himg
            hand_near = []
            hand_objs = []
            hand_pos = []

            for i in range(2):
                if f"hand{i}_obj_class_id_senario" not in info:
                    near_obj = np.array([-1, 10., 0, 0, 0])
                else:
                    near_obj = [info[f"hand{i}_obj_class_id_senario"], info[f"hand{i}_obj_dis"]] + list(info[f"hand{i}_obj_iou"])
                    near_obj = np.array(near_obj)
                hand_near.append(near_obj)
                hand_objs.append(info.get(f"hand{i}_obj_infos", np.zeros((4, 4))))
                hand_pos1 = info.get("hand_keypoints_3d_raw", np.zeros((2, 1, 3)))[i][0]
                hand_pos1 = zed_module.correct_pos(hand_pos1, cam_trans=cam_trans, h1_head_qpos=self.h1_head_qpos)
                hand_pos.append(hand_pos1)
            
            if cam_rot is not None and cam_trans is not None:
                self.zed_pose_msg.data = np.concatenate([cam_rot, cam_trans]).tolist()
                self.zed_pose_pub.publish(self.zed_pose_msg)
            
            # import ipdb; ipdb.set_trace()
            # Type I: send index 0 only
            # self.hand_position_msg.data = np.concatenate(hand_pos).tolist()
            # Type II: send all hand details
            hand_pos_all_info = info.get("hand_keypoints_3d_raw", np.zeros((2, 1, 3)))
            if isinstance(hand_pos_all_info, list):
                hand_pos_all_info = np.concatenate(hand_pos_all_info)
            self.hand_position_msg.data = hand_pos_all_info.flatten().tolist()
            # Send message
            self.hand_position_pub.publish(self.hand_position_msg)

                # hand_pos.append(info[f"hand_keypoints_3d"][i][0])
            head_pos = head_info.get("head_keypoints_3d", np.zeros(3))
            head_pos = zed_module.correct_pos(head_pos, cam_trans=cam_trans, h1_head_qpos=self.h1_head_qpos)
            
            # hand_pos_all_info = info.get("hand_keypoints_3d_raw", np.zeros((2, 1, 3)))
            # print("hand_pos", hand_pos_all_info)
            # print("hand_pos.shape", hand_pos_all_info.shape, np.concatenate(hand_pos_all_info).shape)
            # print("hand_near", hand_near)

            # TODO: add hand_objs
            self.hand_details = [np.concatenate(hand_pos, axis=-1).astype(np.float32), 
                                #  hand_imgs, 
                                 np.concatenate(hand_near, axis=-1).astype(np.float32), 
                                 np.array(head_pos).astype(np.float32)]
        self.last_arm_rots = rots
        self.fid = fid
        
        self.publish_target_pos()

    def publish_target_pos(self):
        msg = ByteMultiArray()
        # pack all infos into one array
        # img: uint8, hand_pos: float32, hand_near: float32
        send_list = [self.body_array] + self.hand_details
        # convert to byte array
        data = b""
        for item in send_list:
            data = data + item.flatten().tobytes()
        # convert to list of one bytes
        msg.data = [data[i:i+1] for i in range(len(data))]
        self.publisher_.publish(msg)
        
        msg = Float32MultiArray()
        msg.data = self.body_array.tolist()
        self.publisher_old.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)


def zed_image_processor(body_shm_info, img_shm_info, svo_file=None, 
                        visualize=False, enable_hand_details=False):
    """ Deprecated Function """
    detect_level="high"
    confidence_threshold = 0.5
    zed_module = ZedModule(svo_file=svo_file, visualize=visualize, 
                            obj_detection=True,
                            detect_level=detect_level, 
                            model_level="medium", auto_replay=True,
                            img_shm_info=img_shm_info)
    hand_detector = HandDetector(zed_module=zed_module, hand_vis_every=-1)
    body_oppo_name, body_shape = body_shm_info
    body_shm = shared_memory.SharedMemory(name=body_oppo_name)
    body_array = np.ndarray(body_shape, dtype=np.float32, buffer=body_shm.buf)

    bodies = 0
    err_cnt = 0
    rots = np.zeros((6, 4))
    # rots[:] = np.nan
    rots[:, 0] = 1
    nan_rot =rots.copy()
    
    all_rots = []
    # all_left_imgs = []

    image_shm_name, image_shm_shape = img_shm_info

    zed_image_shm = shared_memory.SharedMemory(name=image_shm_name)
    zed_image = np.ndarray(image_shm_shape, dtype=np.uint8, buffer=zed_image_shm.buf)

    fid = -1

    fps = 30

    frameperiod = 1.0 / fps
    put_i = 0
    buffer = []

    now = time.time()
    nextframe = now + frameperiod
    last_t = []

    while True:
        last_t.append(time.time())
        last_t = last_t[-30:]
        # if fid % 10 == 0:
        #     print("[Zed] fps", len(last_t)/(last_t[-1] - last_t[0]))

        imgs = zed_module.get_image(render=False)

        zed_module.sync_img_shm(imgs)
        zed_module.detect_obj(render=True)
        if imgs[0] is None:
            break
        fid += 1
        img = imgs[0].get_data()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # img_right = imgs[1].get_data()
        # img_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2BGR)

        # # put image into shared memory
        # zed_image[:] = np.stack([img, img_right], axis=0)
        body_info = zed_module.get_joint_info()
        if len(body_info) != 1:
            err_cnt += 1
            print(f"[Warning] Number of bodies detected is not 1, but {len(body_info)}, with {err_cnt} times.")
        ismiss = True                
        if len(body_info) > 0:
            # track arms
            if detect_level == 'high':
                arm_list = [12, 14, 16, 13, 15, 17]
            elif detect_level == 'medium':
                arm_list = [5, 6, 7, 12, 13, 14]

            for i, body_i in enumerate(arm_list):
                # print("body_info[0][2][body_i]", body_info[0][2][body_i])
                if (body_info[0][2][body_i] >= confidence_threshold and not np.isnan(body_info[0][2][body_i]).any()) or np.isnan(rots[i]).any():
                    rots[i] = body_info[0][1][body_i]
                    
        # track hands
        # TODO: run hand detection in a separate process
        if fid %6 == 0:
        # if False:
            hand_image, hand_data, hand_raw_data, hand_success = self.hand_detector.process_zed_current_img(img, bgr2rgb=False)
            body_array[36:] = hand_data[[0, 1, 4, 6, 8, 10, 12, 13, 16, 18, 20, 22]].flatten() 
            if visualize:
                cv2.imshow("hand_image", hand_image)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        # from quat to 6d
        # ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
        sixd_rot = R.from_quat(rots).as_matrix()[..., :2, :].reshape(-1, 6)  # (6, 6)
        body_array[:36] = sixd_rot.flatten()
        
        if enable_hand_details:
            zed_module.detect_obj(render=obj_vis, call_mode="sync")
            info = zed_module.get_hand_details(img)
            
            # image
            res = 128
            himg = np.zeros((2, res, res, 3), dtype=np.uint8)
            
            for i in range(2):
                # print("resize", (info["hand_imgs"][i].shape))
                if info["hand_imgs"][i] is None or info["hand_imgs"][i].shape[0] == 0 or info["hand_imgs"][i].shape[1] == 0:
                    himg[i] = np.zeros((res, res, 3), dtype=np.uint8)
                else:
                    himg[i] = cv2.resize(info["hand_imgs"][i], (res, res))
                
            himg = himg.reshape(2*res, res, 3)
            
            # pack all infos into one array
            # img: uint8, hand_pos: float32, hand_near: float32
        
        if now < nextframe:
            time.sleep(nextframe - now)
            now = time.time()
        nextframe += frameperiod

    body_shm.close()
    cv2.destroyAllWindows()


def main(args):
    # NOTE: must use spawn method for multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    oppo_body_memory_name = "oppo_motion_shared_memory"
    body_motion_shape = (12 + 36, )
    body_oppo_motion_shm = shared_memory.SharedMemory(
        name=oppo_body_memory_name,
        create=True,
        size=get_shm_size(body_motion_shape, dtype=np.float32),
    )
    img_shm_name = "zed_img"
    img_shape = (2, 720, 1280, 3)
    zed_img_shm = shared_memory.SharedMemory(
        name=img_shm_name,
        create=True,
        size=get_shm_size(img_shape, dtype=np.uint8),
    )
    
    reset_event = mp.Event()

    if args.use_camera:
        svo_file = None
    else:
        svo_file = "test.svo2"
    
    processes = []

    try:
        processes.append(mp.Process(
            target=buffer_processer,
            args=(
                {
                    "reset_event": reset_event,
                    "topic_name": "human_pose/with_hand",
                    "img_shm_info": (img_shm_name, img_shape),
                    "body_shm_info": (oppo_body_memory_name, body_motion_shape),
                    "svo_file": svo_file,
                    "visualize": args.vis,
                    "scenario": args.scenario,
                },
            ),
        ))
        [p.start() for p in processes]
        [p.join() for p in processes]

    except KeyboardInterrupt:
        [p.terminate() for p in processes]
        [p.join() for p in processes]

    finally:
        body_oppo_motion_shm.close()
        body_oppo_motion_shm.unlink()
        zed_img_shm.close()
        zed_img_shm.unlink()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_camera", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--scenario", type=str, default="dining", choices=["dining", "office", "0", "1"], 
                        help="react scenario, 0: dining, 1: office")
    args = parser.parse_args()
    
    main(args)
