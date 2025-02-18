# ref: https://www.stereolabs.com/docs/object-detection/custom-od

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep
import multiprocessing as mp
from multiprocessing import shared_memory
from ultralytics.utils.torch_utils import scale_img

YOLO_CLASS_NAME = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
    12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
CLASS_MAP={
    "dining": {
        0: "bottle", 1: "bowl", 2: "cake", 3: "cup", 4:"person", 5:"sponge",
        # 0: "bottle", 1: "bowl", 2: "cake", 3:"cell phone", 4: "cup", 5:"person", 6:"sponge"
    },
    "dining2": {
        0: "bottle", 1: "bowl", 2: "cake", 3:"cell phone", 4: "cup", 5:"person", 6:"sponge",
    },
    "office": {
        0: 'book', 1: 'cap', 2: 'cell phone', 3: 'hatstand', 4: 'lamp', 5: 'laptop', 6: 'person', 7: 'stamp'
    },
}

YOLO_CLASS_NAME = {
    0: "bottle", 1: "bowl", 2: "cake", 3:"cell phone", 4: "cup", 5:"person", 6:"sponge"
}
YOLO_CLASS_NAME = {
    0: 'book', 1: 'cap', 2: 'cell phone', 3: 'hatstand', 4: 'lamp', 5: 'laptop', 6: 'person', 7: 'stamp'
}
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from ros_utils.simple_rpc import SimpleRPC

lock = Lock()
run_signal = False
exit_signal = False

YOLO_MODEL_DIR = os.path.join(os.path.dirname(__file__), "yolo_ckpts")

def draw_hand_obj(fig_path, hand_pos, objs, scenario="dining"):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hand_color = ["r", "g"]
    obj_color = {"bottle": "b", "cup": "c", "bowl": "m", "plate": "y"}
    # for hand, pos in hand_pos.items():
    for i, pos in enumerate(hand_pos):
        if pos is None:
            continue
        # p = np.array([p[1] for p in pos])
        # p = p.reshape((-1, 3))
        lbl = f"hand_{i}"
        p = pos
        ax.scatter(p[ 0], p[1], p[2], label=lbl, color=hand_color[i], alpha=0.2)
        # ax.plot(p[:, 0], p[:, 1], p[:, 2], label=hand, color=hand_color[hand])
    YOLO_CLASS_NAME = CLASS_MAP[scenario]
    for i, obj in enumerate(objs.object_list):
        if obj is None:
            continue
        p = obj.position
        cid = obj.raw_label
        cname = YOLO_CLASS_NAME[cid]
        lbl = f"obj_{cname}"
        ax.scatter(p[ 0], p[1], p[2], label=lbl, color=obj_color.get(cname, "black"), alpha=0.2)
        # ax.plot(p[:, 0], p[:, 1], p[:, 2], label=obj, color=obj_color[obj])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    # xlim
    ax.set_xlim(0.5, 2)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.savefig(fig_path)

def xywh2abcd(xywh, shape_scale=(1,1)):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * shape_scale[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * shape_scale[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * shape_scale[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * shape_scale[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

# NOTE: this is the list of removed classes
FILTER_CLS = ['person', 'sink', 'table', 'chair', "dining table", "cat", "mouse"]

def detections_to_custom_box(detections, shape_scale, scenario="dining"):
    YOLO_CLASS_NAME = CLASS_MAP[scenario]
    output = []
    
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.unique_object_id = sl.generate_unique_id()
        obj.bounding_box_2d = xywh2abcd(xywh, shape_scale)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        if YOLO_CLASS_NAME[int(det.cls[0])] not in FILTER_CLS:
            output.append(obj)
    return output


def yolo_process(yolo_kwargs, rpc):
    weights = yolo_kwargs['weights']
    img_size = yolo_kwargs.get('img_size', 416)

    img_size = 640
    conf_thres = yolo_kwargs.get('conf_thres', 0.01)
    iou_thres = yolo_kwargs.get('iou_thres', 0.45)
    model = YOLO(weights)
    rpc.set_caller(False)
    # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 
    # 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
    # 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
    # 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
    # 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
    # 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    # 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
    # 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
    # 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    # 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
    # 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
    # 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    # 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
    # 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    det = None
    try:
        while True:
            rpc.wait()
            if rpc.is_quit():
                break
            
            image_net = rpc.buffer['image_np'].copy()
            torch_mode = False
            if torch_mode:
                img = torch.from_numpy(image_net[:1]).permute(0, 3, 1, 2).float()/255.
                img = img[:, [2,1,0]]
                img = scale_img(img, 0.5)
                img = img.cuda()
                # https://docs.ultralytics.com/modes/predict/#video-suffixes
                pred = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, device="cuda:0", verbose=False)
            else:
                pred = model.predict(image_net[0], save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, device="cuda:0", verbose=False)
            det = pred[0].cpu().numpy().boxes
            # detections = detections_to_custom_box(det, image_net)
            rpc.buffer['obj_list'].put(det)
            rpc.response()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("error!!!!!", e)
    
    
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45, scenario="dining"):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            print("det", det)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net, scenario)
            lock.release()
            run_signal = False
        sleep(0.01)

def enable_obj_detection(zed):
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters(instance_module_id=1)
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = False
    obj_param.enable_segmentation = False  # designed to give person pixel mask with internal OD
    zed.enable_object_detection(obj_param)


def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres, "scenario":opt.scenario})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    # if opt.svo is not None:
        # input_type.set_from_svo_file(opt.svo)
    input_type.set_from_svo_file("../demo/front.svo2")

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False  # designed to give person pixel mask with internal OD
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


def test_yolo():
    # numpy	np.zeros((640,1280,3))	np.ndarray	HWC format with BGR channels uint8 (0-255).
    # torch	torch.zeros(16,3,320,640)	torch.Tensor	BCHW format with RGB channels float32 (0.0-1.0).

    # weights = yolo_kwargs['weights']
    # img_size = yolo_kwargs.get('img_size', 416)
    # conf_thres = yolo_kwargs.get('conf_thres', 0.2)
    # iou_thres = yolo_kwargs.get('iou_thres', 0.45)
    weights = "zed_module/yolo_ckpts/yolo11x_2_tune140.pt"
    img_size = 416
    conf_thres = 0.2
    iou_thres = 0.45

    model = YOLO(weights)
    # rpc.set_caller(False)
    while True:
        # rpc.wait()
        
        # image_net = np.zeros((128, 128, 4), dtype=np.uint8)
        image_net = cv2.imread("img0.jpg")
        # img = cv2.cvtColor(image_net, cv2.COLOR_BGR2RGB)
        img = image_net
        img = cv2.resize(img, (1024, 576))
        print(image_net.shape, img.shape)

        # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()/255.
        # img = img.cuda()
        # https://docs.ultralytics.com/modes/predict/#video-suffixes

        det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, device="cuda:0")[0].cpu().numpy().boxes
        detections = detections_to_custom_box(det, image_net)
        break
        # rpc.buffer['obj_list'].put(detections)
        # rpc.response()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov11m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        # main()
        test_yolo()
