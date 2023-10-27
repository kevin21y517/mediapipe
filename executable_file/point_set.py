# from mp_setting import mp_set


class p_set():


    def __init__(self):
        self.results = None
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
        self.right_eye_outer = None
        self.left_ear = None
        self.right_ear = None
        self.mouth_left = None
        self.mouth_right = None
        self.left_shoulder = None
        self.right_shoulder = None
        self.left_elbow = None
        self.right_elbow = None
        self.left_wrist = None
        self.right_wrist = None
        self.left_pinky = None
        self.right_pinky = None
        self.left_index = None
        self.right_index = None
        self.left_thumb = None
        self.right_thumb = None
        self.left_hip = None
        self.right_hip = None
        self.left_knee = None
        self.right_knee = None
        self.left_ankle = None
        self.right_ankle = None
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
            self.right_eye_outer = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
            self.left_ear = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            self.right_ear = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            self.mouth_left = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT]
            self.mouth_right = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT]
            self.left_shoulder = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            self.right_shoulder = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            self.left_elbow = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            self.right_elbow = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            self.left_wrist = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            self.right_wrist = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            self.left_pinky = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY]
            self.right_pinky = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY]
            self.left_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX]
            self.right_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX]
            self.left_thumb = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB]
            self.right_thumb = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB]
            self.left_hip = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            self.right_hip = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            self.left_knee = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            self.right_knee = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            self.left_ankle = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            self.right_ankle = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            self.left_heel = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
            self.right_heel = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
            self.left_foot_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            self.right_foot_index = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]


    def center_point(self):
        # 计算新原点的坐标，即左髋和右髋坐标的平均值
        self.origin_x = (self.left_hip.x + self.right_hip.x) / 2.0
        self.origin_y = (self.left_hip.y + self.right_hip.y) / 2.0
        self.origin_z = (self.left_hip.z + self.right_hip.z) / 2.0

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
        # 计算左眼外角的歸一化坐標
        self.left_eye_outer = (
            (self.left_eye_outer.x - self.origin_x),
            (self.origin_y - self.left_eye_outer.y),
            (self.left_eye_outer.z)
        )
        # 计算右眼内角的歸一化坐標
        self.right_eye_inner = (
            (self.right_eye_inner.x - self.origin_x),
            (self.origin_y - self.right_eye_inner.y),
            (self.right_eye_inner.z)
        )
        # 计算右眼的歸一化坐標
        self.right_eye = (
            (self.right_eye.x - self.origin_x),
            (self.origin_y - self.right_eye.y),
            (self.right_eye.z)
        )
        # 计算右眼外角的歸一化坐標
        self.right_eye_outer = (
            (self.right_eye_outer.x - self.origin_x),
            (self.origin_y - self.right_eye_outer.y),
            (self.right_eye_outer.z)
        )
        # 计算左耳的歸一化坐標
        self.left_ear = (
            (self.left_ear.x - self.origin_x),
            (self.origin_y - self.left_ear.y),
            (self.left_ear.z)
        )
        # 计算右耳的歸一化坐標
        self.right_ear = (
            (self.right_ear.x - self.origin_x),
            (self.origin_y - self.right_ear.y),
            (self.right_ear.z)
        )
        # 计算左嘴角的歸一化坐標
        self.mouth_left = (
            (self.mouth_left.x - self.origin_x),
            (self.origin_y - self.mouth_left.y),
            (self.mouth_left.z)
        )
        # 计算右嘴角的歸一化坐標
        self.mouth_right = (
            (self.mouth_right.x - self.origin_x),
            (self.origin_y - self.mouth_right.y),
            (self.mouth_right.z)
        )
        # 计算左肩的歸一化坐標
        self.left_shoulder = (
            (self.left_shoulder.x - self.origin_x),
            (self.origin_y - self.left_shoulder.y),
            (self.left_shoulder.z)
        )
        # 计算右肩的歸一化坐標
        self.right_shoulder = (
            (self.right_shoulder.x - self.origin_x),
            (self.origin_y - self.right_shoulder.y),
            (self.right_shoulder.z)
        )
        # 计算左肘的歸一化坐標
        self.left_elbow = (
            (self.left_elbow.x - self.origin_x),
            (self.origin_y - self.left_elbow.y),
            (self.left_elbow.z)
        )
        # 计算右肘的歸一化坐標
        self.right_elbow = (
            (self.right_elbow.x - self.origin_x),
            (self.origin_y - self.right_elbow.y),
            (self.right_elbow.z)
        )
        # 计算左手腕的歸一化坐標
        self.left_wrist = (
            (self.left_wrist.x - self.origin_x),
            (self.origin_y - self.left_wrist.y),
            (self.left_wrist.z)
        )
        # 计算右手腕的歸一化坐標
        self.right_wrist = (
            (self.right_wrist.x - self.origin_x),
            (self.origin_y - self.right_wrist.y),
            (self.right_wrist.z)
        )
        # 计算左小指的歸一化坐標
        self.left_pinky = (
            (self.left_pinky.x - self.origin_x),
            (self.origin_y - self.left_pinky.y),
            (self.left_pinky.z)
        )
        # 计算右小指的歸一化坐標
        self.right_pinky = (
            (self.right_pinky.x - self.origin_x),
            (self.origin_y - self.right_pinky.y),
            (self.right_pinky.z)
        )
        # 计算左食指的歸一化坐標
        self.left_index = (
            (self.left_index.x - self.origin_x),
            (self.origin_y - self.left_index.y),
            (self.left_index.z)
        )
        # 计算右食指的歸一化坐標
        self.right_index = (
            (self.right_index.x - self.origin_x),
            (self.origin_y - self.right_index.y),
            (self.right_index.z)
        )
        # 计算左拇指的歸一化坐標
        self.left_thumb = (
            (self.left_thumb.x - self.origin_x),
            (self.origin_y - self.left_thumb.y),
            (self.left_thumb.z)
        )
        # 计算右拇指的歸一化坐標
        self.right_thumb = (
            (self.right_thumb.x - self.origin_x),
            (self.origin_y - self.right_thumb.y),
            (self.right_thumb.z)
        )
        # 计算左髋的歸一化坐標
        self.left_hip = (
            (self.left_hip.x - self.origin_x),
            (self.origin_y - self.left_hip.y),
            (self.left_hip.z)
        )
        # 计算右髋的歸一化坐標
        self.right_hip = (
            (self.right_hip.x - self.origin_x),
            (self.origin_y - self.right_hip.y),
            (self.right_hip.z)
        )
        # 计算左膝的歸一化坐標
        self.left_knee = (
            (self.left_knee.x - self.origin_x),
            (self.origin_y - self.left_knee.y),
            (self.left_knee.z)
        )
        # 计算右膝的歸一化坐標
        self.right_knee = (
            (self.right_knee.x - self.origin_x),
            (self.origin_y - self.right_knee.y),
            (self.right_knee.z)
        )
        # 计算左脚踝的歸一化坐標
        self.left_ankle = (
            (self.left_ankle.x - self.origin_x),
            (self.origin_y - self.left_ankle.y),
            (self.left_ankle.z)
        )
        # 计算右脚踝的歸一化坐標
        self.right_ankle = (
            (self.right_ankle.x - self.origin_x),
            (self.origin_y - self.right_ankle.y),
            (self.right_ankle.z)
        )
        # 计算左脚跟的歸一化坐標
        self.left_heel = (
            (self.left_heel.x - self.origin_x),
            (self.origin_y - self.left_heel.y),
            (self.left_heel.z)
        )
        # 计算右脚跟的歸一化坐標
        self.right_heel = (
            (self.right_heel.x - self.origin_x),
            (self.origin_y - self.right_heel.y),
            (self.right_heel.z)
        )
        # 计算左脚大拇指的歸一化坐標
        self.left_foot_index = (
            (self.left_foot_index.x - self.origin_x),
            (self.origin_y - self.left_foot_index.y),
            (self.left_foot_index.z)
        )
        # 计算右脚大拇指的歸一化坐標
        self.right_foot_index = (
            (self.right_foot_index.x - self.origin_x),
            (self.origin_y - self.right_foot_index.y),
            (self.right_foot_index.z)
        )


    def point_writing(self):
        if self.results.pose_landmarks is not None:

            pose_data = PoseData(self.results)


            # 修改需要的关键点坐标
            pose_data.keypoints[self.mp_pose.PoseLandmark.NOSE] = {
                "x": self.nose[0],
                "y": self.nose[1],
                "z": self.nose[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_EYE_INNER] = {
                "x": self.left_eye_inner[0],
                "y": self.left_eye_inner[1],
                "z": self.left_eye_inner[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_EYE] = {
                "x": self.left_eye[0],
                "y": self.left_eye[1],
                "z": self.left_eye[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER] = {
                "x": self.left_eye_outer[0],
                "y": self.left_eye_outer[1],
                "z": self.left_eye_outer[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER] = {
                "x": self.right_eye_inner[0],
                "y": self.right_eye_inner[1],
                "z": self.right_eye_inner[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_EYE] = {
                "x": self.right_eye[0],
                "y": self.right_eye[1],
                "z": self.right_eye[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER] = {
                "x": self.right_eye_outer[0],
                "y": self.right_eye_outer[1],
                "z": self.right_eye_outer[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_EAR] = {
                "x": self.left_ear[0],
                "y": self.left_ear[1],
                "z": self.left_ear[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_EAR] = {
                "x": self.right_ear[0],
                "y": self.right_ear[1],
                "z": self.right_ear[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.MOUTH_LEFT] = {
                "x": self.mouth_left[0],
                "y": self.mouth_left[1],
                "z": self.mouth_left[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.MOUTH_RIGHT] = {
                "x": self.mouth_right[0],
                "y": self.mouth_right[1],
                "z": self.mouth_right[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_SHOULDER] = {
                "x": self.left_shoulder[0],
                "y": self.left_shoulder[1],
                "z": self.left_shoulder[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] = {
                "x": self.right_shoulder[0],
                "y": self.right_shoulder[1],
                "z": self.right_shoulder[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_ELBOW] = {
                "x": self.left_elbow[0],
                "y": self.left_elbow[1],
                "z": self.left_elbow[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_ELBOW] = {
                "x": self.right_elbow[0],
                "y": self.right_elbow[1],
                "z": self.right_elbow[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_WRIST] = {
                "x": self.left_wrist[0],
                "y": self.left_wrist[1],
                "z": self.left_wrist[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_WRIST] = {
                "x": self.right_wrist[0],
                "y": self.right_wrist[1],
                "z": self.right_wrist[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_PINKY] = {
                "x": self.left_pinky[0],
                "y": self.left_pinky[1],
                "z": self.left_pinky[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_PINKY] = {
                "x": self.right_pinky[0],
                "y": self.right_pinky[1],
                "z": self.right_pinky[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_INDEX] = {
                "x": self.left_index[0],
                "y": self.left_index[1],
                "z": self.left_index[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_INDEX] = {
                "x": self.right_index[0],
                "y": self.right_index[1],
                "z": self.right_index[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_THUMB] = {
                "x": self.left_thumb[0],
                "y": self.left_thumb[1],
                "z": self.left_thumb[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_THUMB] = {
                "x": self.right_thumb[0],
                "y": self.right_thumb[1],
                "z": self.right_thumb[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_HIP] = {
                "x": self.left_hip[0],
                "y": self.left_hip[1],
                "z": self.left_hip[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_HIP] = {
                "x": self.right_hip[0],
                "y": self.right_hip[1],
                "z": self.right_hip[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_KNEE] = {
                "x": self.left_knee[0],
                "y": self.left_knee[1],
                "z": self.left_knee[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_KNEE] = {
                "x": self.right_knee[0],
                "y": self.right_knee[1],
                "z": self.right_knee[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_ANKLE] = {
                "x": self.left_ankle[0],
                "y": self.left_ankle[1],
                "z": self.left_ankle[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_ANKLE] = {
                "x": self.right_ankle[0],
                "y": self.right_ankle[1],
                "z": self.right_ankle[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_HEEL] = {
                "x": self.left_heel[0],
                "y": self.left_heel[1],
                "z": self.left_heel[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_HEEL] = {
                "x": self.right_heel[0],
                "y": self.right_heel[1],
                "z": self.right_heel[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX] = {
                "x": self.left_foot_index[0],
                "y": self.left_foot_index[1],
                "z": self.left_foot_index[2]
            }
            pose_data.keypoints[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX] = {
                "x": self.right_foot_index[0],
                "y": self.right_foot_index[1],
                "z": self.right_foot_index[2]
            }

            # 将pose_data转换为results
            for landmark, coords in pose_data.keypoints.items():
                self.results.pose_landmarks.landmark[landmark].x = coords["x"]
                self.results.pose_landmarks.landmark[landmark].y = coords["y"]
                self.results.pose_landmarks.landmark[landmark].z = coords["z"]
        return self.results


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


