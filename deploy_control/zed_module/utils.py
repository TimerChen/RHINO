import numpy as np
import torch

def get_hand_index(detect_level):
    # detect_level = self.zed_module.body_detect_level
    if detect_level == 'high':
        hand_list = [[16, 30, 32, 34, 36], [17, 31, 33, 35, 37],]
    elif detect_level == 'medium':
        hand_list = [[7,8,9,10], [14,15,16,17]]
    return hand_list

def get_head_index(detect_level):
    # detect_level = self.zed_module.body_detect_level
    if detect_level == 'high':
        head_idx = 5
    elif detect_level == 'medium':
        head_idx = 26
    return head_idx

def keypoints2bbox(keypoints, enlarge=2):
    shapes = keypoints.shape
    avali = keypoints != -1
    if avali.sum() < 2:
        return None
    keypoints = keypoints[avali].reshape(-1, shapes[-1])
    bbox = np.concatenate([keypoints.min(axis=0), keypoints.max(axis=0)])
    if enlarge != 1:
        bbox = enlarge_box(bbox, scale=enlarge, d=shapes[-1])
    return bbox
def enlarge_box(box, scale=2, d=2):
    b2 = box.copy()
    mid = (b2[d:] + b2[:d]) / 2
    b2[:d] = mid - (mid - b2[:d]) * scale
    b2[d:] = mid + (b2[d:] - mid) * scale
    return b2


def bbox_distance_and_iou(box1, box2):
    """
    计算两个任意维度轴对齐矩形框之间的最小距离和IoU。

    参数:
        box1: (min_coords, max_coords)
            min_coords: 维度为 (d,) 的numpy数组，表示盒子在每个维度上的最小坐标。
            max_coords: 维度为 (d,) 的numpy数组，表示盒子在每个维度上的最大坐标。
        box2: 同box1。

    返回:
        distance: 最小距离（浮点数）
        iou: 交并比（浮点数）
    """
    box1 = box1.reshape(2, -1)
    box2 = box2.reshape(2, -1)
    # 将输入转换为numpy数组
    min1, max1 = np.array(box1[0]), np.array(box1[1])
    min2, max2 = np.array(box2[0]), np.array(box2[1])

    # 检查维度是否一致
    assert min1.shape == max1.shape == min2.shape == max2.shape, "两个盒子的维度必须相同"

    # 计算每个维度上的重叠长度
    overlap = np.maximum(0, np.minimum(max1, max2) - np.maximum(min1, min2))

    # 计算交集体积（高维体积）
    intersection = np.prod(overlap)

    # 计算每个盒子的体积
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)

    # 计算并集体积
    union = volume1 + volume2 - intersection

    # 计算IoU
    iou = intersection / union if union != 0 else 0
    iof1 = intersection / volume1 if volume1 != 0 else 0
    iof2 = intersection / volume2 if volume2 != 0 else 0

    # 计算每个维度上的间隔距离
    gaps = np.maximum(0, np.maximum(min1, min2) - np.minimum(max1, max2))
    
    # 最小距离为欧几里得距离
    distance = np.linalg.norm(gaps)

    return distance, (iou, iof1, iof2)
    
def bbox3d_overlap(hbbox, obj_bbox):
    # hbbox: [x1, y1, z1, x2, y2, z2]
    # obj_bbox: [x1, y1, z1, x2, y2, z2]
    # return: overlap ratio
    if hbbox[0] >= obj_bbox[3] or obj_bbox[0] >= hbbox[3]:
        return 0
    if hbbox[1] >= obj_bbox[4] or obj_bbox[1] >= hbbox[4]:
        return 0
    if hbbox[2] >= obj_bbox[5] or obj_bbox[2] >= hbbox[5]:
        return 0
    inter = np.maximum(0, np.minimum(hbbox[3:], obj_bbox[3:]) - np.maximum(hbbox[:3], obj_bbox[:3]))
    inter_vol = np.prod(inter)
    hbbox_vol = np.prod(hbbox[3:] - hbbox[:3])
    obj_bbox_vol = np.prod(obj_bbox[3:] - obj_bbox[:3])
    return inter_vol / (hbbox_vol), inter_vol / (hbbox_vol + obj_bbox_vol - inter_vol)
    # return 

# def bbox3d_distance(hbbox, obj_bbox):
    
    
def get_oppo_body(bodies):
    body_info = {}
    body_info["body_boxes"] = []
    body_info["body_scores"] = []
    conf = -1
    ret_body = None
    for body in bodies.body_list:
        body_info["body_boxes"].append(np.concatenate([body.bounding_box_2d[0], body.bounding_box_2d[2]]))
        body_info["body_scores"].append(body.confidence)
        if body.confidence > conf:
            ret_body = body
            conf = body.confidence
            # self.zed_module.bodies.body_list = [body]
    # keypoints = self.zed_module.bodies.body_list[0].keypoint_2d
    return ret_body, body_info