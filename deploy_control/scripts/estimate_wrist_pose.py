import cv2
import numpy as np
import os
import pickle

with open("h1_assets/calib.pckl", "rb") as f:
    data = pickle.load(f)
    cMat = data[0]
    dcoeff = data[1]
MARKER_SIZE = 0.046

table_x_dis = 0.256
table_y_dis = 0.500

kPi = 3.14159265358979323846
kPi_2 = 1.57079632679489661923

# H1 is facing world x, up is world z, the left hand of H1 is world y
table_in_world = [
    (np.array([table_x_dis/2, table_y_dis/2, 0]), np.array([0, 0, +kPi_2])), # upper left
    (np.array([table_x_dis/2, -table_y_dis/2, 0]), np.array([0, 0, +kPi_2])), # upper right
    (np.array([-table_x_dis/2, -table_y_dis/2, 0]), np.array([0, 0, +kPi_2])), # lower right
    (np.array([-table_x_dis/2, table_y_dis/2, 0]), np.array([0, 0, +kPi_2])), # lower left
]

box_w = 0.074
box_h = 0.064
box_d = 0.060

# body at the zero pos of SMPL, facing wrist x, up is wrist z, then we know y.. The ring is at the xOz plane, 
# the order of four faces: hand pointing, clockwise. Therefore, for right ring, pointing y-, for left ring, pointing y+
box_in_right = [
    (np.array([0, 0, -box_h/2]), np.array([0, kPi, kPi])), # lid
    (np.array([box_w/2, 0, 0]), np.array([kPi, kPi_2, 0])), # hinge
    (np.array([0, 0, box_h/2]), np.array([0, 0, kPi])), # bottom
    (np.array([-box_w/2, 0, 0]), np.array([kPi, -kPi_2, 0])), # clip
]

box_in_left = [
    (np.array([0, 0, -box_h/2]), np.array([0, kPi, 0])), # lid
    (np.array([-box_w/2, 0, 0]), np.array([0, -kPi_2, 0])), # clip
    (np.array([0, 0, box_h/2]), np.array([0, 0.0, 0])), # bottom
    (np.array([box_w/2, 0, 0]), np.array([0, kPi_2, 0])), # hinge
]
# box_in_left = [
#     (np.array([0, 0, box_h/2]), np.array([0, kPi, 0])), # lid
#     (np.array([-box_w/2, 0, 0]), np.array([0, +kPi_2, 0])), # clip
#     (np.array([0, 0, -box_h/2]), np.array([0, 0.0, 0])), # bottom
#     (np.array([box_w/2, 0, 0]), np.array([0, -kPi_2, 0])), # hinge
# ]

id_to_obj = {
    "robot_right": (1,2,3,4),
    "robot_left": (5,6,7,8),
    "table" : (9, 10, 11, 12),
    "human_right": (13,14,15,16),
    "human_left": (17,18,19,20)
}




# 加载相机
# cap = cv2.VideoCapture(0)

# Aruco字典和参数
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11) # 使用的Aruco标记字典
parameters = cv2.aruco.DetectorParameters_create()
ARUCO_DICT = aruco_dict
ARUCO_PARAM = parameters

def array_to_vec(ndarray:np.ndarray):
    assert ndarray.shape == (3,)
    return np.array([[ndarray[0]], [ndarray[1]], [ndarray[2]]])

def vec_to_array(vec:np.ndarray):
    assert vec.shape == (3, 1)
    return np.array([vec[0][0], vec[1][0], vec[2][0]])


def open_camera(VIDEO_ID, width=2560, height=1440, fps=30):
    cap = cv2.VideoCapture(VIDEO_ID)
    if not isinstance(VIDEO_ID, str):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("VIDEO_ID",
              VIDEO_ID,
              "width",
              cap.get(cv2.CAP_PROP_FRAME_WIDTH),
              "height",
              cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
              "fps",
              cap.get(cv2.CAP_PROP_FPS))
    return cap

def get_wrist_in_world_from_markers_in_camera(
    A_B_rvec, A_B_tvec, 
    A_C_rvec, A_C_tvec, 
    D_C_rvec, D_C_tvec, 
    D_W_rvec, D_W_tvec
):
    # Notations:
    # A for marker on the box; B for the wrist in the center of the box; C for camera; D for marker on the table; W for world originated at the center of table.
    # Require: A_B (meaning pose of A in B coordinate system), A_C, D_C, D_W.
    # Acquire: B_W.
    # tvec: rotation vector; rvec: translation vector, (1,3) array.

    # Convert rotation vectors to rotation matrices
    A_B_rotation_matrix, _ = cv2.Rodrigues(A_B_rvec)
    A_C_rotation_matrix, _ = cv2.Rodrigues(A_C_rvec)
    D_C_rotation_matrix, _ = cv2.Rodrigues(D_C_rvec)
    D_W_rotation_matrix, _ = cv2.Rodrigues(D_W_rvec)


    # Calculate the inverse transformation of D in the C coordinate system
    D_C_rotation_matrix_inv = D_C_rotation_matrix.T
    D_C_tvec_inv = -np.dot(D_C_rotation_matrix_inv, D_C_tvec[0][0])

    # Calculate the rotation matrix and translation vector of C in the world coordinate system
    C_W_rotation_matrix = np.dot(D_W_rotation_matrix, D_C_rotation_matrix_inv)
    C_W_tvec = D_W_tvec + np.dot(D_W_rotation_matrix, D_C_tvec_inv)

    # Calculate the rotation matrix and translation vector of A in the world coordinate system
    A_W_rotation_matrix = np.dot(C_W_rotation_matrix, A_C_rotation_matrix)
    A_W_tvec = C_W_tvec + np.dot(C_W_rotation_matrix, A_C_tvec[0][0])

    # Calculate the inverse transformation of A in the B coordinate system
    A_B_rotation_matrix_inv = A_B_rotation_matrix.T
    A_B_tvec_inv = -np.dot(A_B_rotation_matrix_inv, A_B_tvec)

    # Calculate the rotation matrix and translation vector of B in the world coordinate system
    B_W_rotation_matrix = np.dot(A_W_rotation_matrix, A_B_rotation_matrix_inv)
    B_W_tvec = A_W_tvec + np.dot(A_W_rotation_matrix, A_B_tvec_inv)

    # Convert the rotation matrix back to a rotation vector
    B_W_rvec, _ = cv2.Rodrigues(B_W_rotation_matrix)

    return B_W_rvec, B_W_tvec

    # Output the pose of B in the world coordinate system
    print(f"B in World Coordinates: Translation Vector = {B_W_tvec.flatten()}, Rotation Vector (Rodrigues) = {B_W_rvec.flatten()}")

def detect_single_wirst(wrist_name:str, ids, corners):
    # not detected any marker
    if ids is None or ids.size == 0:
        print("No marker detected")
        return None, None
    
    # not detected the table
    table_marker_id_list = []
    table_marker_world_pose_list = []

    for table_marker_id, table_marker_world_pose in zip(id_to_obj["table"], table_in_world):
        if table_marker_id in ids:
            table_marker_id_list.append(table_marker_id)
            table_marker_world_pose_list.append(table_marker_world_pose)
    if len(table_marker_id_list) == 0:
        print("Table not detected")
        return None, None

    wrist_rvec_list = []
    wrist_tvec_list = []
    

    if "right" in wrist_name:
        box_in_wrist = box_in_right
    elif "left" in wrist_name:
        box_in_wrist = box_in_left
    else:
        raise ValueError("wrist_name should contain 'right' or 'left'")
    

    # detected the table
    box_marker_id_list = []
    for box_marker_id, box_marker_wrist_pose in zip(id_to_obj[wrist_name], box_in_wrist):
        
        if box_marker_id in ids:
            box_marker_id_list.append(box_marker_id)

            for table_marker_id, table_marker_world_pose in zip(table_marker_id_list, table_marker_world_pose_list):
                A_corners = corners[list(ids).index(box_marker_id)]
                D_corners = corners[list(ids).index(table_marker_id)]

                A_C_rvec, A_C_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(A_corners, MARKER_SIZE, cMat, dcoeff)
                D_C_rvec, D_C_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(D_corners, MARKER_SIZE, cMat, dcoeff)

                A_B_rvec, A_B_tvec = box_marker_wrist_pose[1], box_marker_wrist_pose[0]
                D_W_rvec, D_W_tvec = table_marker_world_pose[1], table_marker_world_pose[0]

                wrist_rvec, wrist_tvec = get_wrist_in_world_from_markers_in_camera(
                    A_B_rvec, A_B_tvec, 
                    A_C_rvec, A_C_tvec, 
                    D_C_rvec, D_C_tvec, 
                    D_W_rvec, D_W_tvec
                )
                print("wrist_rvec", wrist_rvec)
                print("wrist_tvec", wrist_tvec)
                # (0,1,2)->(0,2,1)
                # wrist_rvec = np.concatenate([-wrist_rvec[2:3], wrist_rvec[1:2], -wrist_rvec[0:1]], axis=0)
                # print("wrist_rvec2", wrist_rvec, wrist_rvec[:, 2:3])
                wrist_rvec_list.append(wrist_rvec)
                wrist_tvec_list.append(wrist_tvec)
                

    if len(wrist_tvec_list) == 0:
        print(f"{wrist_name} not detected")
        return None, None

    return wrist_rvec_list, wrist_tvec_list

if __name__ == "__main__":
    cid = 0
    cap = open_camera(cid, 2560, 1440, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测Aruco标记
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        for wrist_name in ["robot_right", "robot_left", "human_right", "human_left"]:
            wrist_rvec_list, wrist_tvec_list = detect_single_wirst(wrist_name, ids, corners)

            if wrist_tvec_list is not None:
                wrist_rotation = np.array([rvec for rvec in wrist_rvec_list]).mean(axis=0)
                wrist_translation = np.array([tvec for tvec in wrist_tvec_list]).mean(axis=0)
                
                print(f"{wrist_name} detected pose:")
                print("rotation", wrist_rotation)
                print("translation", wrist_translation)
                
        print("corners", ids)
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.imshow('Aruco Detection', frame[::2, ::2, :])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()