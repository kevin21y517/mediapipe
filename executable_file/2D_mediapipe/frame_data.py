# from point_set import p_set
import cv2

class frame_datas():
    def __init__(self):
        # self.point_set = p_set()
        self.frame_data_list = []  # 用于存储每一帧的数据
        self.frame_data = None
        self.image = None
        self.results = None
        # self.new_results = None
        self.mp_pose =  None

        self.prev_nose = None   # 鼻子
        self.prev_left_shoulder = None # 肩膀
        self.prev_right_shoulder = None # 肩膀
        self.prev_left_elbow = None  # 手肘
        self.prev_right_elbow = None # 手肘
        self.prev_left_wrist = None # 手腕
        self.prev_right_wrist = None # 手腕
        self.prev_left_hip = None   # 臀部
        self.prev_right_hip = None  # 臀部
        self.prev_left_knee = None  # 膝盖
        self.prev_right_knee = None # 膝盖
        self.prev_left_ankle = None # 脚踝
        self.prev_right_ankle = None # 脚踝
        self.prev_left_pinky = None # 小指
        self.prev_right_pinky = None # 小指
        self.prev_left_index = None # 食指
        self.prev_right_index = None # 食指
        self.prev_left_thumb = None # 拇指
        self.prev_right_thumb = None # 拇指
        self.prev_left_heel = None # 脚跟
        self.prev_right_heel = None # 脚跟
        self.prev_left_foot_index = None # 脚趾
        self.prev_right_foot_index = None # 脚趾

    def piont_prev(self, results, mp_pose, image):
        self.image = image
        image_height, image_width, z = image.shape
        self.results = results
        self.mp_pose = mp_pose
        if self.results.pose_landmarks:
            self.prev_nose = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y * image_height))
            self.prev_left_shoulder = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height))
            self.prev_right_shoulder = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height))
            self.prev_left_elbow = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height))
            self.prev_right_elbow = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height))
            self.prev_left_wrist = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height))
            self.prev_right_wrist = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height))
            self.prev_left_hip = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * image_height))
            self.prev_right_hip = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * image_height))
            self.prev_left_knee = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y * image_height))
            self.prev_right_knee = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height))
            self.prev_left_ankle = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height))
            self.prev_right_ankle = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height))
            self.prev_left_pinky = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].y * image_height))
            self.prev_right_pinky = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].y * image_height))
            self.prev_left_index = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].y * image_height))
            self.prev_right_index = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y * image_height))
            self.prev_left_thumb = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y * image_height))
            self.prev_right_thumb = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y * image_height))
            self.prev_left_heel = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y * image_height))
            self.prev_right_heel = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].y * image_height))
            self.prev_left_foot_index = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_height))
            self.prev_right_foot_index = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_height))
        pass

    def image_point(self, results):
        self.results = results

        if self.results.pose_landmarks:
            cv2.putText(self.image, f'Nose (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].z:.2f})',
                            (self.prev_nose[0] + 10, self.prev_nose[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Left Shoulder (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z:.2f})',
                            (self.prev_left_shoulder[0] + 10, self.prev_left_shoulder[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Right Shoulder (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z:.2f})',
                            (self.prev_right_shoulder[0] + 10, self.prev_right_shoulder[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Left Elbow (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].z:.2f})',
                            (self.prev_left_elbow[0] + 10, self.prev_left_elbow[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Right Elbow (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].z:.2f})',
                            (self.prev_right_elbow[0] + 10, self.prev_right_elbow[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Left Wrist (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].z:.2f})',
                            (self.prev_left_wrist[0] + 10, self.prev_left_wrist[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Right Wrist (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].z:.2f})',
                            (self.prev_right_wrist[0] + 10, self.prev_right_wrist[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Left Hip (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z:.2f})',
                            (self.prev_left_hip[0] + 10, self.prev_left_hip[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Right Hip (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z:.2f})',
                            (self.prev_right_hip[0] + 10, self.prev_right_hip[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Left Knee (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].z:.2f})',
                            (self.prev_left_knee[0] + 10, self.prev_left_knee[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'Right Knee (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].z:.2f})',
                            (self.prev_right_knee[0] + 10, self.prev_right_knee[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'Left Ankle (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].z:.2f})',
            #                 (self.prev_left_ankle[0] + 10, self.prev_left_ankle[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'Right Ankle (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z:.2f})',
            #                 (self.prev_right_ankle[0] + 10, self.prev_right_ankle[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'left_pinky (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].z:.2f})',
            #                 (self.prev_left_pinky[0] + 10, self.prev_left_pinky[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'right_pinky (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].z:.2f})',
            #                 (self.prev_right_pinky[0] + 10, self.prev_right_pinky[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'left_index (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].z:.2f})',
            #                 (self.prev_left_index[0] + 10, self.prev_left_index[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'right_index (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].z:.2f})',
            #                 (self.prev_right_index[0] + 10, self.prev_right_index[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'left_thumb (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].z:.2f})',
                            (self.prev_left_thumb[0] + 10, self.prev_left_thumb[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'right_thumb (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].z:.2f})',
                            (self.prev_right_thumb[0] + 10, self.prev_right_thumb[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'left_heel (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].z:.2f})',
            #                 (self.prev_left_heel[0] + 10, self.prev_left_heel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # cv2.putText(self.image, f'right_heel (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].z:.2f})',
            #                 (self.prev_right_heel[0] + 10, self.prev_right_heel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'left_foot_index (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z:.2f})',
                            (self.prev_left_foot_index[0] + 10, self.prev_left_foot_index[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(self.image, f'right_foot_index (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z:.2f})',
                            (self.prev_right_foot_index[0] + 10, self.prev_right_foot_index[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow('MediaPipe Pose', self.image)


    def record_point(self, results):
        self.results = results
        self.frame_data = {}  # 为每一帧创建一个新的字典

        for i in range(32):  # 用你的实际关键点数量替换2
            landmark_name = self.mp_pose.PoseLandmark(i).name
            landmark_value = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark(i).value]
            self.frame_data[landmark_name] = self.landmark_to_dict(landmark_value)

        # 将每一帧的数据添加到帧数据列表中
        self.frame_data_list.append(self.frame_data)
        return self.frame_data


    def landmark_to_dict(self,landmark):
        return {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
        }