"""
For Zed API, see: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1BodyData.html

"""
import os
import sys
import argparse

# sys.path.insert(0, "/home/estar/miniforge3/envs/mocap_py38/lib/python3.8/site-packages")
# sys.path.insert(0, "/home/jxchen/miniconda3/envs/mocap310/lib/python3.10/site-packages")
sys.path.append(os.curdir)
from ros_utils.simple_rpc import SimpleRPC
from zed_module.yolo_detector import enable_obj_detection, yolo_process, \
    YOLO_MODEL_DIR, detections_to_custom_box, CLASS_MAP
from zed_module.utils import keypoints2bbox, bbox_distance_and_iou, get_oppo_body, \
    get_hand_index, enlarge_box, get_head_index
from ros_utils.display_utils import get_shm_size

import multiprocessing as mp
from multiprocessing import shared_memory

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

import pyzed.sl as sl
try:
    import ros_utils.cv_viewer.tracking_viewer as cv_viewer
except:
    print("[Warning] cv_viewer can not load, please add ros_utils into the path.")

ZED_FPS = 60
ZED_CALIB_FILE = "h1_assets/head_state_to_zed_pose.npy"

camera_kwargs = {
    # "VIDEO_ID": VIDEO_ID,
    "camera_scale": 1,
    "fps": 30,
}

CONFIG_FILE_PATH_LEFT = "h1_assets/dex_retargeting/inspire_hand_left.yml"
CONFIG_FILE_PATH_RIGHT = "h1_assets/dex_retargeting/inspire_hand_right.yml"


def enable_positinal_tracking(zed):
    """ref: https://www.stereolabs.com/docs/positional-tracking/using-tracking"""
    """ref: https://www.stereolabs.com/docs/positional-tracking/coordinate-frames#world-frame"""
    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    positional_tracking_parameters.enable_imu_fusion = True
    positional_tracking_parameters.mode = sl.POSITIONAL_TRACKING_MODE.GEN_2
    # positional_tracking_parameters.enable_area_memory = True
    # positional_tracking_parameters.area_file_path = "./scenario1_test.area"
    err = zed.enable_positional_tracking(positional_tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit()

    # enable_spatial_mapping(zed)

def enable_spatial_mapping(zed):
    # Enable spatial mapping
    mapping_parameters = sl.SpatialMappingParameters()
    err = zed.enable_spatial_mapping(mapping_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(-1)

def enable_body_tracking(zed, enable_body_tracking=True, detect_level="low", model_level="low"):
    enable_positinal_tracking(zed)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = True            # Smooth skeleton move
    if model_level == "low":
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    elif model_level == "medium":
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
    elif model_level == "high":
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    else:
        assert False, f"Invalid model_level: {model_level}"
        
    if detect_level == "low":
        body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use
    elif detect_level == "medium":
        body_param.body_format = sl.BODY_FORMAT.BODY_34
    elif detect_level == "high":
        body_param.body_format = sl.BODY_FORMAT.BODY_38
    else:
        assert False, f"Invalid detect_level: {detect_level}"

    body_param.body_selection = sl.BODY_KEYPOINTS_SELECTION.UPPER_BODY

    # Enable Object Detection module
    if enable_body_tracking:
        zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    return body_param, body_runtime_param

def init_zed(fps=60, svo_file=None, 
             async_mode=False, svo_real_time=False):
    zed = sl.Camera()
    # NOTE: see https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1InitParameters.html
    init_params = sl.InitParameters(svo_real_time_mode=svo_real_time and (svo_file is not None),
                                    async_image_retrieval=async_mode)
    if svo_file is not None:
        init_params.set_from_svo_file(svo_file)
    # init_params.set_from_svo_file("~/Code/img2_3dpos/demo/front.svo2")

    init_params.camera_resolution = sl.RESOLUTION.HD720  
    init_params.coordinate_units = sl.UNIT.METER
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    init_params.camera_fps = fps  # Set fps at 60
    # init_params.async_image = async_mode
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    # NOTE: see https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1RuntimeParameters.html
    runtime_params = sl.RuntimeParameters(measure3D_reference_frame=sl.REFERENCE_FRAME.WORLD)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()
    return zed, runtime_params

class ZedModule():
    def __init__(self, svo_file=None, 
                 body_tracking=True, 
                 obj_detection=False,
                 visualize=None, 
                 detect_level="high", model_level="medium",
                 auto_replay=False,
                 img_shm_info=None,
                 async_mode=True,
                 svo_real_time=False,
                 scenario="dining",
                 ):
        self.zed, self.runtime_param = init_zed(svo_file=svo_file, async_mode=async_mode, 
                                                svo_real_time=svo_real_time, )
        if scenario in ["0", "1"]:
            scenario_map = {"0": "dining", "1": "office"}
            scenario = scenario_map[scenario]
        self.scenario = scenario
        self.async_mode = async_mode
        self.svo_real_time = svo_real_time
        self.svo_file = svo_file
        self.auto_replay = auto_replay
        # if body_tracking:
        self.body_tracking = body_tracking
        body_param, body_runtime_param = enable_body_tracking(self.zed, body_tracking, 
                                                              detect_level=detect_level, model_level=model_level)
        self.body_detect_level = detect_level
        self.body_model_level = model_level
        self.body_param = body_param
        self.body_runtime_param = body_runtime_param
        
        self._init_zed_pose_table()
        self.init_transform = None
            
        self.img_shm_info = img_shm_info
        if img_shm_info is not None:
            self.init_img_shm()
            
        self.obj_detection = obj_detection
        if obj_detection:
            self.init_obj_detection()
        
        camera_info = self.zed.get_camera_information()
        display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
        self.camera_info = camera_info
        self.display_resolution = display_resolution
        self.image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                    , display_resolution.height / camera_info.camera_configuration.resolution.height]
        # cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
        self.image = sl.Mat()
        self.image_R = sl.Mat()
        self.ptcloud = sl.Mat()
        self.depth_img = sl.Mat()
        self.depth = sl.Mat()

        self.bodies = sl.Bodies()
        self.sensors_data = sl.SensorsData()
        self.visualize = visualize
        self.show_text = ""
        
    @staticmethod
    def create_zed_img_shm(img_shm_name="zed_img"):
        img_shape = (2, 720, 1280, 3)
        zed_img_shm = shared_memory.SharedMemory(
            name=img_shm_name,
            create=True,
            size=get_shm_size(img_shape, dtype=np.uint8),
        )
        return img_shm_name, img_shape, zed_img_shm
        
    @property
    def total_frames(self):
        return self.zed.get_svo_number_of_frames()
    
    def reset_positional_tracking(self):
        self.zed.reset_positional_tracking(self.init_transform)
    
    def _init_zed_pose_table(self):
        self.head_state_to_zed_pose_table = np.load(ZED_CALIB_FILE)
    
    def correct_pos(self, pos, h1_head_qpos, cam_rot=None, cam_trans=None):
        """Remove the camera shift, and use a table to correct the zed position with head state."""
        if cam_trans is None:
            cam_rot, cam_trans = self.get_it()
        if cam_trans is None:
            print("[Warning] No camera translation found.")
            return pos
        # The followiing two are under zedm frame, zed_pose_from_zed: get from zed msg, zed_pose_from_head: get from zed calibration table.
        zed_pose_from_head = self.head_state_to_zed_pose_table[
            np.argmin(np.sum(np.abs(self.head_state_to_zed_pose_table[:, 0:2] - h1_head_qpos), axis=1))
        ][5:8]
        pos = pos - cam_trans + zed_pose_from_head
        return pos
    
    def start_record(self, record_path):
        svo_filename = record_path + "zed" + ".svo"
        recording_param = sl.RecordingParameters(
            svo_filename, sl.SVO_COMPRESSION_MODE.H264
        )
        err = self.zed.enable_recording(recording_param)
        
    def stop_record(self):
        self.zed.disable_recording()
    
    def jump_to_frame(self, frame_id):
        self.zed.set_svo_position(frame_id)
        
    def reload(self, svo_file):
        self.zed, self.runtime_param = init_zed(svo_file=svo_file, async_mode=self.async_mode, 
                                                svo_real_time=self.svo_real_time, )
        self.svo_file = svo_file
        body_param, body_runtime_param = enable_body_tracking(self.zed, self.body_tracking, 
                                                              detect_level=self.body_detect_level, model_level=self.body_model_level)
        self.body_param = body_param
        self.body_runtime_param = body_runtime_param
        camera_info = self.zed.get_camera_information()
        display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
        self.camera_info = camera_info
        self.display_resolution = display_resolution
        self.image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                    , display_resolution.height / camera_info.camera_configuration.resolution.height]
        if self.obj_detection:
            enable_obj_detection(self.zed)
        
    def init_img_shm(self):
        image_shm_name, image_shm_shape = self.img_shm_info
        self.image_shm = shared_memory.SharedMemory(name=image_shm_name)
        self.image_shm_np = np.ndarray(image_shm_shape, dtype=np.uint8, buffer=self.image_shm.buf)
        
    def init_obj_detection(self):
        enable_obj_detection(self.zed)
        rpc_buffer = {
            # "image_np": self.image_shm_np,
            "image_np": self.img_shm_info,
            "obj_list": mp.Queue(),
        }
        self.detect_rpc = SimpleRPC(buffer=rpc_buffer)
        if self.scenario == "dining":
            weights_path = os.path.join(YOLO_MODEL_DIR, "yolo11x_tune123.pt")
        elif self.scenario == "office":
            weights_path = os.path.join(YOLO_MODEL_DIR, "yolo11x_2_tune140.pt")
        else:
            assert False, f"Invalid scenario: {self.scenario}"
            
        yolo_kwargs = {
            "weights": weights_path,
            "scenario": self.scenario,
            "img_size": 640,
            "conf_thres": 0.2,
            "iou_thres": 0.45,
        }
        self.detect_process = mp.Process(target=yolo_process, args=(yolo_kwargs, self.detect_rpc,))
        self.detect_process.start()
        self.detect_rpc.set_caller(True)
        self.objects = sl.Objects()
        self.zed_obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.zed_obj_runtime_param.detection_confidence_threshold = 1
        
    def filter_failed_obj(self):
        new_list = []
        for obj in self.objects.object_list:
            if not np.isnan(obj.position).any():
                new_list.append(obj)
        
        self.objects = sl.Objects()
        self.objects.object_list = new_list

    def check_positinal_tracking(self):
        tracking_state = self.zed.get_positional_tracking_status()
        area_export_state = self.zed.get_area_export_state()
        # SPATIAL_MAPPING_STATE
        spatial_mapping_state = self.zed.get_spatial_mapping_state()
        print("tracking_state", tracking_state)
        print("area_export_state", area_export_state)
        print("spatial_mapping_state", spatial_mapping_state)

    def save_map(self, map_file):
        self.zed.save_area_map(map_file)
        print("export status", repr(self.zed.get_area_export_state()))

        
    def detect_obj(self, render=False, call_mode="sync"):
        import time
        st = time.time()
        if call_mode == "sync":
            self.detect_rpc.call(no_wait=False)
        elif call_mode == "async":
            self.detect_rpc.call(no_wait=True)
            return None
        elif call_mode == "get":
            self.detect_rpc.wait()
            
        detections = self.detect_rpc.buffer["obj_list"].get()
        # img_shape (2, H, W, 3)
        # detections = detections_to_custom_box(detections, (1280/1024., 720./576, 4))
        # detections = detections_to_custom_box(detections, (1280/640., 720./384, 4))

        detections = detections_to_custom_box(detections, (1., 1., 4), scenario=self.scenario)
        
        self.zed.ingest_custom_box_objects(detections, 1)
        # print("t1", time.time()-st)
        self.zed.retrieve_objects(self.objects, self.zed_obj_runtime_param, 1)
        # print("t2", time.time()-st)
        self.filter_failed_obj()
        
        if (isinstance(render, bool) and render==True) or isinstance(render, str):
            image_left_ocv = self.image.get_data().copy()
            cv_viewer.render_2D_obj(image_left_ocv, self.image_scale, self.objects, is_tracking_on=False, scenario=self.scenario)
            if isinstance(render, str):
                cv2.imwrite(render, image_left_ocv)
            else:
                self.update_cv2(image_left_ocv, "objs")
        elif isinstance(render, np.ndarray):
            cv_viewer.render_2D_obj(render, self.image_scale, self.objects, is_tracking_on=False, scenario=self.scenario)
        
        return self.objects

    def get_sensor(self):
        sensors_data = self.sensors_data
        if self.zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS :
            pass
            quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
            # print(" \t Orientation: [ Ox: {0}, Oy: {1}, Oz {2}, Ow: {3} ]".format(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))

    def get_image(self, render=True, skip_body=False):

        if self.auto_replay and (self.svo_file is not None) and \
            self.zed.get_svo_position() >= self.total_frames-2:
            self.zed.set_svo_position(0)

        if self.zed.grab(self.runtime_param) == sl.ERROR_CODE.SUCCESS:
            # image, bodies = self.image, self.bodies
            
            zed = self.zed
            image = self.image
            if self.init_transform is None:
                pose = sl.Pose()
                zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                self.init_transform = pose.pose_data()
                print("!!!!init_transform!!!!", self.init_transform.m)

            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
            zed.retrieve_image(self.image_R, sl.VIEW.RIGHT, sl.MEM.CPU, self.display_resolution)

            # zed.retrieve_image(self.depth_img, sl.VIEW.DEPTH, )
            # zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH, )
            # zed.retrieve_measure(self.ptcloud, sl.MEASURE.XYZ, )
            # ptcloud_np = np.array(self.ptcloud.get_data())
            # depth_np = np.array(self.depth.get_data())
            # depth_img_np = np.array(self.depth_img.get_data())
            # print("ptcloud", ptcloud_np.shape, depth_np.shape, depth_img_np.shape)
            # print("value", ptcloud_np[500:510, 500:510, ])
            if not skip_body:
                self.detect_body(render=render)
        
        else:
            return [None, None]

        ret = [self.image, self.image_R]
        if self.img_shm_info is not None:
            self.sync_img_shm(ret)
        return ret
    
    def detect_body(self, render=True):
        bodies = self.bodies
        body_param = self.body_param
        # Retrieve bodies
        if self.body_tracking:
            self.zed.retrieve_bodies(bodies, self.body_runtime_param)
        
        # cv2.putText(image_left_ocv, "11", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow("ZED | 2D View", image_left_ocv)
        if isinstance(render, bool) and render:
            image_left_ocv = self.image.get_data().copy()
            cv_viewer.render_2D_body(image_left_ocv, self.image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            self.update_cv2(image_left_ocv, "body")
        elif isinstance(render, np.ndarray):
            cv_viewer.render_2D_body(render, self.image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
    
    def sync_img_shm(self, imgs):
        if self.img_shm_info is None:
            return
        
        if imgs[0] is None:
            return
        # fid += 1
        img = imgs[0].get_data()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_right = imgs[1].get_data()
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2BGR)

        cv2.imwrite("4yoloimg.jpg", img)

        # put image into shared memory
        self.image_shm_np[:] = np.stack([img, img_right], axis=0)
        
    # def init_visualizer(self):
    #     if not isinstance(self.visualize, str) and self.visualize.endswith(".avi"):
    #         # h264
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         out = cv2.VideoWriter(self.visualize, fourcc, 20.0, (1280, 720))
    #         self.visual_writer = out
    
    def update_cv2(self, image_left_ocv, suffix=""):
        # if self.visualize==True:
        show_text = self.show_text
        cv2.putText(image_left_ocv, show_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(f"ZED {suffix}", image_left_ocv)
        key_wait = 10
        key = cv2.waitKey(key_wait)
        if key == 113: # for 'q' key
            print("Exiting...")
            exit()
            # break
        if key == 109: # for 'm' key
            if (key_wait>0):
                print("Pause")
                key_wait = 0 
            else : 
                print("Restart")
                key_wait = 10 
        # elif self.visualize.endswith(".avi"):
        #     # cv2.imwrite(self.visualize, image_left_ocv)
        #     image_left_ocv = cv2.resize(image_left_ocv, (1280, 720))
        #     self.visual_writer.write(image_left_ocv)

    # def get_it(self):
    #     zed = self.zed
    #     camera_pose = sl.Pose()
    #     py_translation = sl.Translation()
    #     tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
    #     tracking_status = zed.get_positional_tracking_status()

    #     #Get rotation and translation and displays it
    #     if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
    #         rotation = camera_pose.get_rotation_vector()
    #         translation = camera_pose.get_translation(py_translation)
    #         # text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
    #         # text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))

    #     pose_data = camera_pose.pose_data(sl.Transform())

    def get_it(self):
        zed = self.zed
        camera_pose = sl.Pose()
        py_translation = sl.Translation()
        tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
        tracking_status = zed.get_positional_tracking_status()

        #Get rotation and translation and displays it
        rot, trans = None, None
        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
            rotation = camera_pose.get_rotation_vector()
            translation = camera_pose.get_translation(py_translation).get()
            text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
            text_translation = str((round(translation[0], 2), round(translation[1], 2), round(translation[2], 2)))
            # print("[get_it]", "Rotation: ", text_rotation, "Translation: ", text_translation)
            rot = rotation
            trans = translation

        pose_data = camera_pose.pose_data(sl.Transform())
        return rot, trans

    def transform_pos(self, objs):
        def transform_pose(camera_pose, obj_position, l2r_translation):
            transform_ = sl.Transform()
            transform_.set_identity()
            # Translate the tracking frame by tx along the X axis
            for i in range(3):
                transform_[i, 3] = obj_position[i]
            # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
            
            # transform2_ = sl.Transform()
            # transform2_.set_identity()
            # transform2_[0, 3] = tx
            transform_inv = sl.Transform()
            transform_inv.init_matrix(l2r_translation)
            transform_inv.inverse()
            pose = transform_inv * camera_pose * transform_
            return pose

        zed = self.zed
        camera_pose = sl.Pose()
        # Retrieve and transform the pose data into a new frame located at the center of the camera
        tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        pos_list = []
        # print("tx", zed.get_camera_information().camera_configuration.calibration_parameters.stereo_transform)
        l2r = zed.get_camera_information().camera_configuration.calibration_parameters.stereo_transform
        for obj in objs:
            pose = transform_pose(camera_pose.pose_data(sl.Transform()), obj, l2r)
            pos = [pose[0, 3], pose[1, 3], pose[2, 3]]
            # print("get position", pose.get_translation().get())
            
            pos_list.append(pos)
        
        return pos_list
    
    def get_joint_info(self):
        """
        18 arm: 5 6 7, 2 3 4
        34 arm: 5 6 7, 12 13 14
        38 arm: 12 14 16, 13 15 17
        """
        bodies = self.bodies
        body_infos = []
        for body in bodies.body_list:
            body_infos.append((body.keypoint_2d, body.local_orientation_per_joint, body.keypoint_confidence))
        return body_infos

    def get_wrist_pos(self):
        bodies = self.bodies
        body_param = self.body_param
        ret = []
        for body in bodies.body_list:
            if body_param.body_format == sl.BODY_FORMAT.BODY_18:
                left_wrist = body.keypoint[7]
                right_wrist = body.keypoint[4]
                # ret = ret + left_wrist.tolist() + right_wrist.tolist()
            elif body_param.body_format == sl.BODY_FORMAT.BODY_34:
                left_wrist = body.keypoint[7]
                right_wrist = body.keypoint[14]
            elif body_param.body_format == sl.BODY_FORMAT.BODY_38:
                left_wrist = body.keypoint[16]
                right_wrist = body.keypoint[17]
                left_wrist_rot = body.local_position_per_joint[16]
                right_wrist_rot = body.local_position_per_joint[17]
            
            # left_wrist = left_wrist[[2, 0, 1]]
            # right_wrist = right_wrist[[2, 0, 1]]
            # left_wrist[0:2] = -left_wrist[0:2]
            # right_wrist[0:2] = -right_wrist[0:2]
            self.show_text = f"L: {left_wrist[0]:.2f}, {left_wrist[1]:.2f}, {left_wrist[2]:.2f}"
            self.show_text += f" R: {right_wrist[0]:.2f}, {right_wrist[1]:.2f}, {right_wrist[2]:.2f}\n"
            left_wrist, right_wrist = self.transform_pos([left_wrist, right_wrist])
            # ret = ret + left_wrist.tolist() + right_wrist.tolist()
            ret = ret + left_wrist + right_wrist
            self.show_text += f"L: {left_wrist[0]:.2f}, {left_wrist[1]:.2f}, {left_wrist[2]:.2f}"
            self.show_text += f" R: {right_wrist[0]:.2f}, {right_wrist[1]:.2f}, {right_wrist[2]:.2f}"
        return ret
    
    def get_nearst_obj(self, hand_info):
        """ Get the nearest object to the hand

        Args:
            hand_info (dict): hand info
        Return:
            hand_info with nearest object info
        """
        handbbox = hand_info["hand_bbox_3d_raw"]
        
        senario_obj_id = {
            "dining": ["bottle", "cup", "bowl", "cake", "cell phone"],
            "dining2": ["bottle", "cup", "bowl", "cell phone"],
            "office": ["cap", "book", "stamp", "cell phone"],
        }
        obj_id = senario_obj_id[self.scenario]
        YOLO_CLASS_NAME = CLASS_MAP[self.scenario]
        
        rets = {}
        obj_list = []
        # for hbbox in handbbox:
        for i, hbbox in enumerate(handbbox):
            obj_infos = np.array([np.array([10.,0,0,0]) for _ in range(len(obj_id))])
            
            if hbbox is None:
                # rets.append(None)
                continue
            ret = {}
            for obj in self.objects.object_list:
                # get overlap and distance of 2 3d bbox
                object_id = obj.id # Get the object id
                object_position = obj.position # Get the object position
                object_velocity = obj.velocity # Get the object velocity
                object_tracking_state = obj.tracking_state # Get the tracking state of the object
                class_id = obj.raw_label
                class_name = YOLO_CLASS_NAME[class_id]
                obj_bbox = obj.bounding_box
                obj_bbox = np.concatenate([np.min(obj_bbox, axis=0), np.max(obj_bbox, axis=0)])
                dist, ious = bbox_distance_and_iou(hbbox, obj_bbox)
                
                if class_name not in obj_id:
                    continue
                
                obj_list.append((class_id, class_name, dist, ious, obj_bbox))
                if dist < ret.get("dis", 10.) or (dist == 0 and ious[1] > ret.get("obj_iou", (0.,0,0))[1]):
                    ret[f"class_id"] = class_id
                    ret[f"class_name"] = class_name
                    ret[f"class_id_senario"] = obj_id.index(class_name)
                    ret[f"bbox"] = obj_bbox
                    ret[f"dis"] = dist
                    ret[f"iou"] = ious
                    
                if class_name in obj_id:
                    idx = obj_id.index(class_name)
                    if dist < obj_infos[idx][0] or (dist == 0 and ious[1] > obj_infos[idx][1]):
                        obj_infos[idx] = np.array([dist, ious[0], ious[1], ious[2]])
                
            for k,v in ret.items():
                hand_info[f"hand{i}_obj_{k}"] = v
            hand_info[f"hand{i}_obj_list"] = obj_list
            hand_info[f"hand{i}_obj_infos"] = obj_infos
        return hand_info
    
    def get_hand_details(self, img):
        """
        Return: Hand kepoints, Hand bbox(2d + 3d), bbox img
        """
        body, body_info = get_oppo_body(self.bodies)
        if body is None:
            return {}
        keypoints = body.keypoint_2d
        hand_list = get_hand_index(self.body_detect_level)
        # hand wrist keypoints
        hand_keypoints_2d = [body.keypoint_2d[side_hand] for side_hand in hand_list]
        hand_keypoints_3d_raw = [body.keypoint[side_hand] for side_hand in hand_list]
        
        # hand_keypoints_3d = [np.array(self.transform_pos(kp)) for kp in hand_keypoints_3d_raw]
        hand_keypoints_3d = hand_keypoints_3d_raw
        
        # hand bbox
        hand_bbox_2d = [keypoints2bbox(kps, enlarge=2) for kps in hand_keypoints_2d]
        hand_bbox_3d_raw = [keypoints2bbox(kps, enlarge=2) for kps in hand_keypoints_3d_raw]
        hand_bbox_3d = [keypoints2bbox(kps, enlarge=2) for kps in hand_keypoints_3d]
        # hand bbox img
        imgs = []
        for bbox in hand_bbox_2d:
            if bbox is not None:
                # bimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                ibbox = [int(x)for x in bbox]
                ibbox[0] = np.clip(ibbox[0], 0, img.shape[1])
                ibbox[1] = np.clip(ibbox[1], 0, img.shape[0])
                ibbox[2] = np.clip(ibbox[2], 0, img.shape[1])
                ibbox[3] = np.clip(ibbox[3], 0, img.shape[0])
                bimg = img[ibbox[1]:ibbox[3], ibbox[0]:ibbox[2]]
                imgs.append(bimg)
            else:
                imgs.append(None)        
        
        info = {
            "hand_keypoints_2d": hand_keypoints_2d,
            "hand_keypoints_3d": hand_keypoints_3d,
            "hand_keypoints_3d_raw": hand_keypoints_3d_raw,
            "hand_bbox_2d": hand_bbox_2d,
            "hand_bbox_3d": hand_bbox_3d,
            "hand_bbox_3d_raw": hand_bbox_3d_raw,
            "hand_imgs": imgs,
        }
        if self.obj_detection:
            info = self.get_nearst_obj(info)
        return info


    def get_head_details(self):
        """
        Return: Head kepoints, Head bbox(2d + 3d)
        """
        body, body_info = get_oppo_body(self.bodies)
        if body is None:
            return {}
        head_idx = get_head_index(self.body_detect_level)
        head_keypoint_2d = body.keypoint_2d[head_idx]
        head_keypoint_3d = body.keypoint[head_idx]
        
        head_bbox_2d = keypoints2bbox(head_keypoint_2d, enlarge=2)
        head_bbox_3d = keypoints2bbox(head_keypoint_3d, enlarge=2)
        
        body_pos = body.position 
        body_bbox = body.bounding_box
        body_bbox_2d = body.bounding_box_2d  
        confidence = body.confidence
        head_pos = body.head_position
        
        info = {
            "head_keypoints_2d": head_keypoint_2d,
            "head_keypoints_3d": head_keypoint_3d,
            "head_bbox_2d": head_bbox_2d,
            "head_bbox_3d": head_bbox_3d,
            "body_pos": body_pos,
            "body_bbox": body_bbox,
            "body_bbox_2d": body_bbox_2d,
            "confidence": confidence,
            "head_pos": head_pos,
        }
        return info
        

    def __del__(self):
        if self.obj_detection:
            self.detect_process.terminate()
            self.detect_process.join()
            
        self.image.free(sl.MEM.CPU)
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.close()
        cv2.destroyAllWindows()
