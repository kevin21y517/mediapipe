import cv2
import mediapipe as mp
import pykinect_azure as pykinect

class pyk_set():

    def __init__(self):
        pykinect.initialize_libraries()
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        self.device = pykinect.start_device(config=self.device_config)

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
    def pyk_open(self):
        capture = self.device.update()
        self.success, self.image = capture.get_color_image()
        return self.success

    def pyk_setting(self):
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
