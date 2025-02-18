import os
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch

sys.path.append(os.curdir)
from zed_module.utils import get_oppo_body, get_hand_index, keypoints2bbox, enlarge_box

sys.path.append('../')
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

import matplotlib.pyplot as plt

RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
BROWN = (19, 69, 139)
LIGHT_PURPLE = (0.68235294,  0.61960784,  0.84705882)
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
LIGHT_RED = (0.83921569,  0.37254902,  0.37254902)
LIGHT_GREEN = (0.54901961,  0.74117647,  0.41960784)
LIGHT_YELLOW = (0.99215686,  0.68235294,  0.38039216)
LIGHT_BROWN = (0.80392157,  0.52156863,  0.24705882)
COLOR_MAP = {
    0: RED,
    1: BLUE,
    2: YELLOW,
    3: GREEN,
    4: PURPLE,
    5: BROWN,
    6: LIGHT_PURPLE,
    7: LIGHT_BLUE,
    8: LIGHT_RED,
    9: LIGHT_GREEN,
    10: LIGHT_YELLOW,
    11: LIGHT_BROWN,
}

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


def hand6to12(hand_pos):
    assert(len(hand_pos) == 6)
    target_pos = np.array([hand_pos[0], hand_pos[1], hand_pos[1] * 1.4, hand_pos[1] * 0.6, 
                hand_pos[2], hand_pos[2], hand_pos[3], hand_pos[3], hand_pos[4], hand_pos[4], hand_pos[5], hand_pos[5]])
    return target_pos


class HandDetector:
    def __init__(
        self,
        zed_module=None,
        hand_vis_every=100,
        use_hsv=False,
        rule_track=False,
    ):
        # Setup HaMeR model
        # download_models(CACHE_DIR_HAMER)
        self.hand_model, self.hand_model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.hand_model = self.hand_model.to(self.device)
        self.hand_model.eval()
        self.use_hsv = use_hsv
        self.rule_track = rule_track

        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        # cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_l_100ep.py'
        # cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_b_100ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        # detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_l/f328021305/model_final_1a9f28.pkl"
        # detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        self.start_steps = 20
        self.reset_track_info()

        self.cpm = ViTPoseModel(self.device)
        self.last_left_hand, self.last_right_hand = np.zeros(12), np.zeros(12)        
        self.last_left_raw_hand, self.last_right_raw_hand = np.zeros(45), np.zeros(45)
        if hand_vis_every > 0:
            self.hand_renderer = Renderer(self.hand_model_cfg, faces=self.hand_model.mano.faces)
        else:
            self.hand_renderer = None
        self.zed_module = zed_module
        self.hand_vis_every = hand_vis_every
        self.hand_track_pos = None
        self.hand_track_history_size = []
        
    def draw_box(self, img, bboxes, label_text=None):
        img = np.ascontiguousarray(img)
        for i, bbox in enumerate(bboxes):
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            # p1 = bbox[0]
            # p2 = bbox[1]
            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
            if label_text is not None:
                cv2.putText(img, f"{label_text[i]}", (p1[0]+10, p1[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return img
    
    def draw_points(self, img, points, put_score=True):
        img = np.ascontiguousarray(img)
        for p in points:
            p1 = (int(p[0]), int(p[1]))
            p2 = (int(p[0])+2, int(p[1])+2)
            cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
            if put_score:
                text = f"{int(p[2]*10)}"
                cv2.putText(img, text, (p1[0]+10, p1[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # cv2.imwrite("./test_img.png", img)
        return img
    
    def reset_track_info(self):
        self.target_ids = []
        self.last_ori = {}
        self.last_angle = {}
        self.last_verts = {}
        self.last_cam_t = {}
        self.last_is_right = {}
        self.step = -1
        
        self.hand_track_pos = None
        self.hand_track_history_size = []

    def process_zed_current_img(self, img, bgr2rgb=False):
        # use zed's bodies to detect hand boxes
        if len(self.zed_module.bodies.body_list) == 0:
            return (
                img,
                np.concatenate((self.last_left_hand, self.last_right_hand)),
                np.concatenate((self.last_left_raw_hand, self.last_right_raw_hand)),
                False,
                # np.ones((2, 3)) * 5.
            )
        
        # select the body with the highest confidence
        body, body_info = get_oppo_body(self.zed_module.bodies)
        keypoints = body.keypoint_2d
        hand_list = get_hand_index(self.zed_module.body_detect_level)
        
        lbbox = keypoints2bbox(keypoints[hand_list[0]])
        rbbox = keypoints2bbox(keypoints[hand_list[1]])
        bboxes = []
        is_right = []
        if lbbox is not None:
            bboxes.append(lbbox)
            is_right.append(0)
        if rbbox is not None:
            bboxes.append(rbbox)
            is_right.append(1)
        # print("bboxes", bboxes)
        # img = self.draw_box(img, bboxes)
        # if len(bboxes) == 0:
        #     cv2.imwrite("./detect_failed_img.png", img)
        # else:
        #     cv2.imwrite("./test_img.png", img)
        # exit()
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.process_hand_by_boxes(img, bboxes, is_right, body_info)
    
    def preprocess_blue_hand(self, input_image, ):
        if self.use_hsv:
            hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            # lower_gray = np.array([100, 43, 43])
            # upper_gray = np.array([124, 255, 255])
            # mask = cv2.inRange(hsv, lower_gray, upper_gray)
            # cond = mask > 0
            # cond = mask == 0
            # hsv[..., 0] = np.where(cond, 0, hsv[..., 0])
            # hsv[..., 1] = np.where(cond, 0, hsv[..., 1])
            # hsv[..., 2] = np.where(cond, 0, hsv[..., 2])
            
            # hsv[..., 0] = (hsv[..., 0] - 94//2) % 180
            hsv[..., 0] = np.mod(hsv[..., 0].astype(np.int32) +(- 160//2), 180).astype(np.uint8)
            # hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int32) -20, 0, 250).astype(np.uint8)
            input_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # directly change blue and red channel
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        return input_image
    
    def mask_blue_hand(self, input_image):
        # input_image = input_image.copy()
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([100, 43, 43])
        upper_gray = np.array([124, 255, 255])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        input_image[mask==0] = 0
        return input_image[:, :, ::-1]
    
    def process_image(self, input_image, skip_detect=False, bgr2rgb=False, blue_hand=True, rule_mode=True, first_person=False):

        from motion_utils.humanoid_hand_detector import get_humanoid_hand_bbox
        if (not rule_mode) or (not blue_hand):
            if rule_mode:
                print("[WARNING] Rule mode is only available for blue hand detection")
            return self.process_image_old(input_image, skip_detect, bgr2rgb, blue_hand)
        
        # cv2.imwrite("./test_img.png", input_image)
        # exit()
        bboxes_hand = get_humanoid_hand_bbox(input_image, first_person=first_person, scale=30, track_pos=self.hand_track_pos)
        
        img = self.preprocess_blue_hand(input_image.copy())

        if bboxes_hand is None or len(bboxes_hand) < 2:
            bboxes_hand = []
        elif self.rule_track:
            self.hand_track_pos = []
            history_sz = self.hand_track_history_size
            for bbox in bboxes_hand:
                sz = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if len(history_sz) < 10:
                    history_sz.append(sz)
                else:
                    sz = np.mean(history_sz)
                    
                self.hand_track_pos.append((sz, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)))

        body_info = {}
        bboxes = []
        is_right = []
        for i, bbox in enumerate(bboxes_hand):
            bboxes.append(np.array([bbox[1], bbox[0], bbox[3], bbox[2]]))
            cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 2)
            # always in order of left, right
            is_right.append(i)
        body_info["body_boxes"] = bboxes_hand
        body_info["body_scores"] = [1.0, 1.0]
        # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        # print(bboxes)
        # cv2.imshow("hand_bbox", img)
        # cv2.waitKey(1)
        return self.process_hand_by_boxes(img, bboxes, is_right, body_info)

    def process_hand_by_boxes(self, img, bboxes, is_right, body_info):
        step = self.step
        
        if self.hand_vis_every < 0:
            self.hand_model.skip_mano = True
        else:
            self.hand_model.skip_mano = (step % self.hand_vis_every != 0)
        # print("hand", type(self.hand_model), step, self.hand_vis_every, step % self.hand_vis_every, self.hand_model.skip_mano)
        
        self.step += 1
        if len(bboxes) == 0:
            return (
                # cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR),
                img,
                np.concatenate((self.last_left_hand, self.last_right_hand)),
                np.concatenate((self.last_left_raw_hand, self.last_right_raw_hand)),
                False,
            )

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
            
        dataset = ViTDetDataset(self.hand_model_cfg, img, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
           
        all_verts = []
        all_cam_t = []
        all_right = []
        all_mesh_color = []
        all_ori = []
        left_qpos, right_qpos = None, None
         
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.hand_model(batch)
                
            batch_size = batch['img'].shape[0]  # hand nums
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.hand_model_cfg.EXTRA.FOCAL_LENGTH / self.hand_model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            for n in range(batch_size):
                person_id = int(batch['personid'][n])
                if step < self.start_steps and person_id not in self.target_ids:
                    self.target_ids.append(person_id)
                params = out['pred_pose_params'][n].detach().cpu().numpy()  # (15, 3, 3)
                axis_angle = R.from_matrix(params).as_rotvec()  # (15, 3)
                
                thumb_yaw = np.clip(axis_angle[12, [0,2]].mean() * 1.5, 0, 1)
                thumb_pitch = np.clip(axis_angle[13:15, 0].mean() * 2.5, 0, 1)
                index = np.clip(axis_angle[0:3, 2].mean(), 0, 1)
                middle = np.clip(axis_angle[3:6, 2].mean(), 0, 1)
                ring = np.clip(axis_angle[9:12, 2].mean(), 0, 1)
                pinky = np.clip(axis_angle[6:9, 2].mean() * 1.2, 0, 1)
                target_pos = np.array([thumb_yaw, thumb_pitch, thumb_pitch * 1.4, thumb_pitch * 0.6, 
                                       index, index, middle, middle, ring, ring, pinky, pinky])

                ori = out['global_orient'][n][0].detach().cpu().numpy()  # (3, 3)
                # ori_2x2 = ori[:2, :2]
                # angle = np.arctan2(ori_2x2[1, 0], ori_2x2[0, 0])
                # if angle > 0:
                #     continue
                r = R.from_matrix(ori)
                # quaternion = r.as_quat()
                quaternion_vectors = r.apply(np.eye(3))
                x_angle = np.arccos(np.dot(quaternion_vectors[0], [1, 0, 0]) / (np.linalg.norm(quaternion_vectors[0]) * np.linalg.norm([1, 0, 0])))
                y_angle = np.arccos(np.dot(quaternion_vectors[1], [0, 1, 0]) / (np.linalg.norm(quaternion_vectors[0]) * np.linalg.norm([0, 1, 0])))
                z_angle = np.arccos(np.dot(quaternion_vectors[2], [0, 0, 1]) / (np.linalg.norm(quaternion_vectors[0]) * np.linalg.norm([0, 0, 1])))
                is_right = batch['right'][n].cpu().numpy()
                cam_t = pred_cam_t_full[n]

                if step >= self.start_steps and person_id not in self.target_ids:
                    continue
                diff = 0.4
                # disable angle check
                diff = 3.1415926 * 1.2
                if 'pred_vertices' in out:
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    all_verts.append(verts)
                if step < self.start_steps or (np.abs(x_angle - self.last_angle[person_id][0]) < diff and np.abs(y_angle - self.last_angle[person_id][1]) < diff and np.abs(z_angle - self.last_angle[person_id][2]) < diff):
                    self.last_angle[person_id] = [x_angle, y_angle, z_angle]
                    self.last_ori[person_id] = ori
                    # self.last_verts[person_id] = verts
                    self.last_cam_t[person_id] = cam_t
                    self.last_is_right[person_id] = is_right
                    if is_right:
                        self.last_right_hand = target_pos
                        right_qpos = target_pos
                        self.last_right_raw_hand = axis_angle.flatten()
                        right_raw_hand = axis_angle.flatten()
                    else:
                        self.last_left_hand = target_pos
                        left_qpos = target_pos
                        self.last_left_raw_hand = axis_angle.flatten()
                        left_raw_hand = axis_angle.flatten()
                
                # all_verts.append(self.last_verts[person_id])
                all_cam_t.append(self.last_cam_t[person_id])
                all_right.append(self.last_is_right[person_id])
                all_ori.append([self.last_ori[person_id], person_id, self.last_angle[person_id]])
                all_mesh_color.append(COLOR_MAP[person_id])
        
        # hand_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        # print("no pred_vertices", len(all_verts))
        hand_image = img
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=all_mesh_color,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            ) 
            cam_view = self.hand_renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img.astype(np.float32)[:,:,::-1]/255.0  # bgr2rgb
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel (transparency)
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            hand_image = 255*input_img_overlay[:, :, ::-1]

            start_x, start_y = 10, 10
            box_size = 200
            for ([ori, person, angle], color) in zip(all_ori, all_mesh_color):
                r = R.from_matrix(ori)
                # Create a blank image to draw the orientation
                ori_img = np.ones((box_size, box_size, 3), dtype=np.uint8) * 255
                # Draw the orientation as lines
                center = (box_size // 2, box_size // 2)
                scale = box_size // 2
                # Draw the quaternion as vectors
                quaternion_vectors = r.apply(np.eye(3))
                draw_vectors = r.apply(np.eye(3)) * scale
                for vec, col in zip(draw_vectors, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    end_point = (int(center[0] + vec[0]), int(center[1] + vec[1]))
                    cv2.line(ori_img, center, end_point, col, 2)
                
                # put text on the right up corner
                text = f"{person}:{angle[0]:.2f}, {angle[1]:.2f}, {angle[2]:.2f}"
                # compute the
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = box_size - text_size[0] - 10
                text_y = 10 + text_size[1]
                cv2.putText(ori_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

                # Determine the position to place the orientation image on hand_image
                pos_x = start_x
                pos_y = start_y
                hand_image[pos_y:pos_y + box_size, pos_x:pos_x + box_size] = ori_img
                # Update the position for the next orientation
                start_y += box_size + 10  # Adjust spacing as needed

            # hand_image.dtype = np.uint8
            hand_image = hand_image.astype(np.uint8)
        
        # draw hand boxes
        hand_image = self.draw_box(hand_image, bboxes)
        # draw frame
        cv2.putText(hand_image, f"{step}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # draw body boxes
        if "body_boxes" in body_info:
            hand_image = self.draw_box(hand_image, body_info["body_boxes"], [f"{s:.3f}" for s in body_info["body_scores"]])
            # hand_image = self.draw_points(hand_image, body_info["body_boxes"][:, :2])

        success = True
        if left_qpos is None:
            left_qpos = self.last_left_hand
            left_raw_hand = self.last_left_raw_hand
            success = False
        if right_qpos is None:
            right_qpos = self.last_right_hand
            right_raw_hand = self.last_right_raw_hand
            success = False
                    
        return (
            hand_image,
            np.concatenate((left_qpos, right_qpos)),
            np.concatenate((left_raw_hand, right_raw_hand)),
            success,
            # all_cam_t,
        )