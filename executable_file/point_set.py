# from mp_setting import mp_set

class p_set():


    def __init__(self):
        self.results = None
        self.new_results = None
        self.mp_pose = None
        self.origin = None
        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

        self.nose = None
        self.left_eye_inner = None
        self.left_eye = None
        self.left_eye_outer = None
        self.right_eye_inner = None
        self.right_eye = None
        self.left_hip = None
        self.right_hip = None
        self.left_shoulder = None
        self.right_shoulder = None
        self.left_elbow = None
        self.right_elbow = None
        self.left_wrist = None
        self.right_wrist = None
        self.left_knee = None
        self.right_knee = None
        self.left_ankle = None
        self.right_ankle = None
        self.left_pinky = None
        self.right_pinky = None
        self.left_index = None
        self.right_index = None
        self.left_thumb = None
        self.right_thumb = None
        self.left_heel = None
        self.right_heel = None
        self.left_foot_index = None
        self.right_foot_index = None



    def results_set(self,results,mp_poses):
        self.results = results
        self.mp_pose = mp_poses


    def log_key_point(self):
        if self.results.pose_landmarks is not None:
            self.nose = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            self.left_eye_inner = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER]
            self.left_eye = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
            self.left_eye_outer = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER]
            self.right_eye_inner = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER]
            self.right_eye = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]
            self.left_hip = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            self.right_hip = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            self.left_shoulder = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            self.right_shoulder = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            self.left_elbow = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            self.right_elbow = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            self.left_wrist = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            self.right_wrist = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            self.left_knee = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            self.right_knee = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            self.left_ankle = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            self.right_ankle = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            self.left_pinky = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY]
            self.right_pinky = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY]
            self.left_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX]
            self.right_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX]
            self.left_thumb = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB]
            self.right_thumb = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB]
            self.left_heel = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
            self.right_heel = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
            self.left_foot_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            self.right_foot_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]


    def center_point(self):
        # 计算新原点的坐标，即左髋和右髋坐标的平均值
        self.origin_x = (self.left_hip.x + self.right_hip.x)/2.0
        self.origin_y = (self.left_hip.y + self.right_hip.y)/2.0
        self.origin_z = (self.left_hip.z + self.right_hip.z)/2.0

        # 新原点的坐标即为新的原点
        self.origin = (self.origin_x, self.origin_y, self.origin_z)
        pass

    def new_point(self):
        # 计算鼻子的歸一化坐標
        self.nose = (
            (self.nose.x - self.origin_x),
            (self.origin_y - self.nose.y),
            (self.nose.z)
        )
        # 计算左眼内角的歸一化坐標
        self.left_eye_inner = (
            (self.left_eye_inner.x - self.origin_x),
            (self.origin_y - self.left_eye_inner.y),
            (self.left_eye_inner.z)
        )
        # 计算左眼的歸一化坐標
        self.left_eye = (
            (self.left_eye.x - self.origin_x),
            (self.origin_y - self.left_eye.y),
            (self.left_eye.z)
        )
        pass


    def point_writing(self):
        # if self.results.pose_landmarks is not None:
        #     pose_data = PoseData(self.results)

        #     # 修改需要的关键点，以鼻子为例
        #     pose_data.keypoints[self.mp_pose.PoseLandmark.NOSE] = {
        #         "x": self.nose[0],
        #         "y": self.nose[1],
        #         "z": self.nose[2]
        #     }
        #     # 添加其他需要修改的关键点
        #     self.new_results = self.results
        #     # 将pose_data转换为results
        #     for landmark, coords in pose_data.keypoints.items():
        #         self.new_results.pose_landmarks.landmark[landmark].x = coords["x"]
        #         self.new_results.pose_landmarks.landmark[landmark].y = coords["y"]
        #         self.new_results.pose_landmarks.landmark[landmark].z = coords["z"]
        return self.new_results, self.mp_pose


class PoseData:
        def __init__(self, results):
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            self.keypoints = {}
            if results.pose_landmarks:
                for landmark, point in zip(mp_pose.PoseLandmark, results.pose_landmarks.landmark):
                    self.keypoints[landmark] = {
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                    }


