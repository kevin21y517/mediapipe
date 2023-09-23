import cv2
import mediapipe as mp
import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

prev_nose_coords = None
prev_left_shoulder_coords = None
prev_right_shoulder_coords = None
prev_left_knee_coords = None
prev_right_knee_coords = None

fig = plt.figure()
ax = fig.add_subplot(121)
ax_3d = fig.add_subplot(122, projection='3d')
pose_data = []

def process_pose_estimation(cap, run_time):
    start_time = time.time()
    output_timer = time.time()
    # 获取当前日期和时间
    current_datetime = datetime.datetime.now()
    # 将datetime对象转换为字符串
    formatted_time_for_filename = current_datetime.strftime("%Y-%m-%d_%H%M%S")

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:

        out =  save_video(formatted_time_for_filename)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("忽略空的攝像頭幀.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # 将帧写入输出视频
            out.write(image)

            if results.pose_landmarks:
                key_point,absolute_coordinates=log_key_point(results,image)

                current_time = time.time()
                output_delay = 3.0

                if current_time - output_timer > output_delay:
                    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    frame_data=record_point(key_point,formatted_datetime,absolute_coordinates)
                    pose_data.append(frame_data)
                    output_timer = current_time


            if results.pose_landmarks:
                three_dimensional_model(image,results,ax,ax_3d)
                image_point(image,key_point,absolute_coordinates)



            # 如果按下 ESC 键或计时器达到10秒，则退出循环
            if cv2.waitKey(1) == 27 or (start_time > 100 and time.time() - start_time >= run_time):
                # 在程序结束时创建独立的JSON文件
                json_data(formatted_time_for_filename)
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def save_video(formatted_time_for_filename):
    video_folder = "video"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 创建一个名字基于当前时间的唯一文件名
    output_filename = os.path.join(video_folder, f"output_{formatted_time_for_filename}.avi")
    out = cv2.VideoWriter(output_filename, fourcc, 5.0, (640, 480))  #偵率
    return(out)


def log_key_point(results,image):
        z_max=1
        z_min=-1

        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        image_height, image_width, _ = image.shape

        nose_coordinates = (((nose_landmark.x * image_width - nose_landmark.x * image_width) / (image_width / 2)), ((nose_landmark.y * image_height - nose_landmark.y * image_height) / (image_height / 2)),((nose_landmark.z - nose_landmark.z) / (z_max - z_min)))
        left_shoulder_coordinates = (((left_shoulder.x * image_width - nose_landmark.x * image_width) / (image_width / 2)), ((nose_landmark.y * image_height - left_shoulder.y * image_height) / (image_height / 2)),((left_shoulder.z - nose_landmark.z) / (z_max - z_min)))
        right_shoulder_coordinates = (((right_shoulder.x * image_width - nose_landmark.x * image_width) / (image_width / 2)), ((nose_landmark.y * image_height - right_shoulder.y * image_height) / (image_height / 2)),((right_shoulder.z - nose_landmark.z) / (z_max - z_min)))
        left_knee_coordinates = (((left_knee.x * image_width - nose_landmark.x * image_width) / (image_width / 2)), ((nose_landmark.y * image_height - left_knee.y * image_height) / (image_height / 2)),((left_knee.z - nose_landmark.z) / (z_max - z_min)))
        right_knee_coordinates = (((right_knee.x * image_width - nose_landmark.x * image_width) / (image_width / 2)), ((nose_landmark.y * image_height - right_knee.y * image_height) / (image_height / 2)),((right_knee.z - nose_landmark.z) / (z_max - z_min)))

        key_point=(left_shoulder,right_shoulder,left_knee,right_knee,nose_landmark,image_height, image_width)
        absolute_coordinates=(nose_coordinates,left_shoulder_coordinates,right_shoulder_coordinates,left_knee_coordinates,right_knee_coordinates)

        return(key_point,absolute_coordinates)


def record_point(key_point,formatted_datetime,absolute_coordinates):
    left_shoulder,right_shoulder,left_knee,right_knee,nose_landmark,image_height, image_width=key_point
    nose_coordinates,left_shoulder_coordinates,right_shoulder_coordinates,left_knee_coordinates,right_knee_coordinates=absolute_coordinates

    # 将当前帧的数据添加到列表中
    frame_data = {
        "日期": formatted_datetime,
        "鼻子": {
            "X": round(nose_coordinates[0], 2),
            "Y": round(nose_coordinates[1], 2),
            "Z": round(nose_coordinates[2], 2)
        },
        "左肩膀": {
            "X": round(left_shoulder_coordinates[0], 2),
            "Y": round(left_shoulder_coordinates[1], 2),
            "Z": round(left_shoulder_coordinates[2], 2)
        },
        "右肩膀": {
            "X": round(right_shoulder_coordinates[0], 2),
            "Y": round(right_shoulder_coordinates[1], 2),
            "Z": round(right_shoulder_coordinates[2], 2)
        },
        "左膝盖": {
            "X": round(left_knee_coordinates[0], 2),
            "Y": round(left_knee_coordinates[1], 2),
            "Z": round(left_knee_coordinates[2], 2)
        },
        "右膝盖": {
            "X": round(right_knee_coordinates[0], 2),
            "Y": round(right_knee_coordinates[1], 2),
            "Z": round(right_knee_coordinates[2], 2)
        }
        # 添加其他坐标点
    }
    return frame_data


def image_point(image,key_point,absolute_coordinates):
    left_shoulder,right_shoulder,left_knee,right_knee,nose_landmark,image_height, image_width=key_point
    nose_coordinates,left_shoulder_coordinates,right_shoulder_coordinates,left_knee_coordinates,right_knee_coordinates=absolute_coordinates

    prev_nose_coordinates = (int(nose_landmark.x * image_width), int(nose_landmark.y * image_height))
    prev_left_shoulder_coordinates = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
    prev_right_shoulder_coordinates = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
    prev_left_knee_coordinates = (int(left_knee.x * image_width), int(left_knee.y * image_height))
    prev_right_knee_coordinates = (int(right_knee.x * image_width), int(right_knee.y * image_height))

    if nose_coordinates is not None:
        cv2.putText(image, f'Nose (X,Y,Z): ({nose_coordinates[0]:.2f}, {nose_coordinates[1]:.2f}, {nose_coordinates[2]:.2f})',
                        (prev_nose_coordinates[0] + 10, prev_nose_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(image, f'Left Shoulder (X,Y,Z): ({left_shoulder_coordinates[0]:.2f}, {left_shoulder_coordinates[1]:.2f}, {left_shoulder_coordinates[2]:.2f})',
                        (prev_left_shoulder_coordinates[0] + 10, prev_left_shoulder_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(image, f'Right Shoulder (X,Y,Z): ({right_shoulder_coordinates[0]:.2f}, {right_shoulder_coordinates[1]:.2f}, {right_shoulder_coordinates[2]:.2f})',
                        (prev_right_shoulder_coordinates[0] + 10, prev_right_shoulder_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(image, f'Left Knee (X,Y,Z): ({left_knee_coordinates[0]:.2f}, {left_knee_coordinates[1]:.2f}, {left_knee_coordinates[2]:.2f})',
                        (prev_left_knee_coordinates[0] + 10, prev_left_knee_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(image, f'Right Knee (X,Y,Z): ({right_knee_coordinates[0]:.2f}, {right_knee_coordinates[1]:.2f}, {right_knee_coordinates[2]:.2f})',
                        (prev_right_knee_coordinates[0] + 10, prev_right_knee_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imshow('MediaPipe Pose', image)


def three_dimensional_model(image,results,ax,ax_3d):

    x = [lm.x * image.shape[1] for lm in results.pose_landmarks.landmark]
    y = [lm.y * image.shape[0] for lm in results.pose_landmarks.landmark]
    ax.clear()

    # 垂直翻轉Y坐標以確保頭部朝上
    y_flipped = [image.shape[0] - y_coord for y_coord in y]

    ax.scatter(x, y_flipped, c='b', marker='o')  # 使用y_flipped

    # 繪製連接線和其他標籤
    connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (10, 9),
    (12,11),
    (12,14),
    (14,16),
    (16,22),
    (16,20),
    (16,18),
    (20,18),
    (11,13),
    (13,15),
    (15,21),
    (15,19),
    (15,17),
    (19,17),
    (12,24),
    (11,23),
    (24,23),
    (24,26),
    (26,28),
    (28,32),
    (32,30),
    (30,28),
    (23,25),
    (25,27),
    (27,29),
    (29,31),
    (31,27),
    ]
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        ax.plot([x[start_idx], x[end_idx]], [y_flipped[start_idx], y_flipped[end_idx]], c='r')  # 使用y_flipped

    ax.set_xlabel('x')
    ax.set_ylabel('y')  # Y軸仍然為Y軸
    plt.pause(0.01)

    # 3D散點圖
    ax_3d.clear()
    z = [-lm.z for lm in results.pose_landmarks.landmark]
    ax_3d.scatter(x, z, y_flipped, c='b', marker='o')  # 將y_flipped和z的位置互換

    # 繪製連接線
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        ax_3d.plot([x[start_idx], x[end_idx]], [z[start_idx], z[end_idx]], [y_flipped[start_idx], y_flipped[end_idx]], c='r')  # 交換z和y_flipped的位置

    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('z')  # 將Y軸改為Z軸
    ax_3d.set_zlabel('y')  # 將Z軸改為Y軸
    plt.pause(0.01)

def json_data(formatted_time_for_filename):
    # 定义要保存 JSON 文件的文件夹路径
    json_folder = "json"

    # 确保文件夹存在，如果不存在就创建它
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    # 创建完整的文件路径
    output_filename = os.path.join(json_folder, f"pose_data_{formatted_time_for_filename}.json")
    with open(output_filename, "w", encoding='utf-8') as json_file:
        json.dump(pose_data, json_file, indent=4, ensure_ascii=False)
