# from point_set import p_set
import cv2

class frame_datas():
    def __init__(self):
        # self.point_set = p_set()
        self.frame_data_list = []  # 用于存储每一帧的数据
        self.frame_data = None
        self.image = None
        self.results = None
        self.new_results = None
        self.mp_pose =  None


    def image_point(self, results, new_results, mp_pose, image):
        self.results = results
        self.new_results = new_results
        self.mp_pose = mp_pose
        self.image = image
        image_height, image_width, z = image.shape

        if self.results.pose_landmarks:
            prev_nose_coordinates = (int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x * image_width), int(self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y * image_height))
            # nose_text = f'Nose (X,Y,Z): (
            #     {self.results.pose_landmarks.landmark[self.mp_pose.NOSE.x]:.2f},
            #     {self.results.pose_landmarks.landmark[self.mp_pose.NOSE.y]:.2f},
            #     {self.results.pose_landmarks.landmark[self.mp_pose.NOSE.z]:.2f})'
            # cv2.putText(self.image, nose_text, (prev_nose_coordinates[0] + 10, prev_nose_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.putText(self.image, f'Nose (X,Y,Z): ({self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y:.2f},{self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].z:.2f})',
                            (prev_nose_coordinates[0] + 10, prev_nose_coordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow('MediaPipe Pose', self.image)


    def record_point(self, results, mp_pose):
        # self.new_results = new_results
        self.mp_pose = mp_pose
        self.results = results

        self.frame_data = {}  # 为每一帧创建一个新的字典

        for i in range(32):  # 用你的实际关键点数量替换2
            landmark_name = self.mp_pose.PoseLandmark(i).name
            landmark_value = self.results.pose_landmarks.landmark[self.mp_pose.PoseLandmark(i).value]
            self.frame_data[landmark_name] = self.landmark_to_dict(landmark_value)

        # 将每一帧的数据添加到帧数据列表中
        self.frame_data_list.append(self.frame_data)

        # self.frame_data = {
        #     "鼻子": {
        #     "X": round(self.point_set.nose_coordinates[0], 2),
        #     "Y": round(self.point_set.nose_coordinates[1], 2),
        #     "Z": round(self.point_set.nose_coordinates[2], 2)
        # },
        #     "左肩": {
        #     "X": round(self.point_set.left_shoulder_coordinates[0], 2),
        #     "Y": round(self.point_set.left_shoulder_coordinates[1], 2),
        #     "Z": round(self.point_set.left_shoulder_coordinates[2], 2)
        # },
        #     "右肩": {
        #     "X": round(self.point_set.right_shoulder_coordinates[0], 2),
        #     "Y": round(self.point_set.right_shoulder_coordinates[1], 2),
        #     "Z": round(self.point_set.right_shoulder_coordinates[2], 2)
        # },
        # }
        return self.frame_data


    def landmark_to_dict(self,landmark):
        return {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
        }