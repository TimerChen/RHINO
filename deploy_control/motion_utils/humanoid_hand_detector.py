import numpy as np
import cv2
import subprocess

def mask_blue_hand(input_image, scale=50):
    # input_image = input_image.copy()
    # input_image = input_image[::scale, ::scale]
    input_image = cv2.resize(input_image, (input_image.shape[1]//scale, input_image.shape[0]//scale), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    # lower_gray = np.array([100, 43, 43])
    lower_gray = np.array([100, 43, 100])
    upper_gray = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    input_image[mask==0] = 0
    return input_image[:, :, ::-1], mask

def find_contours(image_mask):
    # image_mask = image_mask.copy()
    # image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    # image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
    image_mask2 = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_mask2, 10, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # return contours
    image_mask = cv2.drawContours(image_mask, contours, -1, (0, 255, 0), 3)
    return image_mask

def flood_fill(vis_mask, x, y):
    # mask = mask.copy()
    h, w = vis_mask.shape
    # vis_mask = np.zeros_like(mask)
    q = [(x, y)]
    bbox = [x, y, x, y]
    cnt = 0
    x0, y0 = x, y
    while q:
        x, y = q.pop(0)
        cnt += 1
        vis_mask[x, y] = 0
        # update bbox
        bbox[0] = min(bbox[0], x)
        bbox[1] = min(bbox[1], y)
        bbox[2] = max(bbox[2], x)
        bbox[3] = max(bbox[3], y)
        # print("flooding...", x0, y0, vis_mask[x0: x0+10, y0: y0+10])

        append = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for xx, yy in append:
            if xx < 0 or xx >= h or yy < 0 or yy >= w:
                continue
            if vis_mask[xx, yy] == 0:
                continue
            q.append((xx, yy))
    return bbox, cnt

def find_connected_area(mask):
    vis_mask = mask.copy()
    bbox_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if vis_mask[i, j]:
                bbox, cnt = flood_fill(vis_mask, i, j)
                bsize = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bbox_list.append((cnt, bbox))
    sorted_bbox = sorted(bbox_list, key=lambda x: x[0], reverse=True)
    return sorted_bbox

def check_line(line):
    breaks = 0
    print("line", line.shape)
    for i in range(1, len(line)):
        if line[i] != line[i-1]:
            breaks += 1
    return breaks

def bbox_detect_area(mask):
    _, mid_res_mask = mask_blue_hand(box_img, 10)
    if np.mean(mid_res_mask) < 0.3:
        return False
    return True

def bbox_detect_line(mask, bbox):
    # _, mask1= mask_blue_hand(frame1, 1)
    # frame1[mask1==0] = 0
    box_img = frame0[bbox[0]*scale:bbox[2]*scale:10, bbox[1]*scale:bbox[3]*scale:10]
    _, mid_res_mask = mask_blue_hand(box_img, 1)
    
    breaks1 = check_line(mid_res_mask[box_img.shape[0] // 2, :])
    breaks2 = check_line(mid_res_mask[:, box_img.shape[1] // 2,])
    if breaks1 > 3 or breaks2 > 3:
        # continue
        return False
    # print("mask", mid_res_mask, box_img.shape)
    mid_res_mask = mid_res_mask.astype(np.float32) / 255.
    return True

def get_humanoid_hand_bbox(frame, first_person=True, scale=20):
    """
    Get the bounding box of the humanoid hand
    First person: True for camera_human.mp4, False for camera_humanoid.mp4(3rd person perspective)
    """
    # frame0 = frame.copy()
    # frame1 = frame.copy()
    frame, mask= mask_blue_hand(frame, scale)
    frame = np.ascontiguousarray(frame)
    bboxes = find_connected_area(mask)
    if first_person:
        bboxes = sorted(bboxes[:2], key=lambda x: x[1], reverse=False)
    else:
        bboxes = sorted(bboxes[:2], key=lambda x: x[1], reverse=True)
    ret = []
    for bbox in bboxes[:2]:
        bbox = bbox[1]
        
        bbox[0] -= 1
        bbox[1] -= 1
        bbox[2] += 2
        bbox[3] += 2
        bbox = [b*scale for b in bbox]
        ret.append(bbox)

    return ret
    
if __name__ == "__main__":    
    # p = "/home/jxchen/Code/webcam2motion/react_data_1129/spread_hand/20241129-201042/camera_humanoid.mp4"
    p = "/home/jxchen/Code/webcam2motion/react_data_1129/spread_hand/20241129-201042/camera_humanoid.mp4"
    # p = "react_data_1129/photo/20241129-200544/camera_humanoid.mp4"

    # open the video
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        print("Error: Could not open video.")
        # sys.exit(-1)
        exit(-1)

    conf_thres = 0.5
    iou_thres = 0.5
    while True:
        ret, frame = cap.read()
        frame0 = frame.copy()
        if not ret:
            break
        bboxes = get_humanoid_hand_bbox(frame, first_person=True, scale=30)
        # for bbox in bboxes:
        bbox = bboxes[0]
        cv2.rectangle(frame0, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
        bbox = bboxes[1]
        cv2.rectangle(frame0, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 255, 255), 2)
        frame0 = frame0[::2, ::2]
        # frame1 = frame1[::2, ::2]
        cv2.imshow("frame", frame0)
        # model.predict(frame, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, device="cuda:0", verbose=False, show=True)
        # results = model(frame)
        # print(results.xyxy[0])
        # cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    