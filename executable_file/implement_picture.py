import cv2
from mp_setting import mp_set
from frame_data import frame_datas
import mediapipe as mp
import json
import os


class picture_receiver:
    def __init__(self):
        self.mp_set = mp_set()
        self.frame_datas = frame_datas()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results= None
        self.image = cv2.imread('img/xin_c200cm_h0cm.JPG')
        self.frame = None


    def detectPose(self):
        self.pose = self.mp_pose.Pose()
        self.results = self.pose.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            self.image,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        self.frame = self.frame_data.record_point()
        self.pose_data.append(self.frame)
        self.json_data()


    def json_data(self):
        # 定义要保存 JSON 文件的文件夹路径
        json_folder = "json"

        # 确保文件夹存在，如果不存在就创建它
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)

        # 创建完整的文件路径
        output_filename = os.path.join(json_folder, f"pose_data.json")
        with open(output_filename, "w", encoding='utf-8') as json_file:
            json.dump(self.pose_data, json_file, indent=4, ensure_ascii=False)