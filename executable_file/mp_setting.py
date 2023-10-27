import cv2
import mediapipe as mp




class mp_set():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1)
        self.results= None

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.results_face= None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.success = None
        self.image = None



    def image_setting(self):
        self.success, self.image = self.cap.read()
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            self.image,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())
        return self.image, self.results, self.mp_pose

    def picture_set(self):
        self.image = cv2.imread('img/xin_c200cm_h0cm.JPG')
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            self.image,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())
        return self.image, self.results, self.mp_pose

    def face_image_setting(self):
        self.success, self.image = self.cap.read()
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results_face = self.face.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            self.image,
            self.results.multi_face_landmarks[0],
            self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_contours_style())
        return self.image, self.results_face, self.mp_face_mesh


