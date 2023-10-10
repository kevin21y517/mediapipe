from views import *
import cv2

image = cv2.imread('D:/Kevin_mediapipe/mediapipe/img/xin_c200cm_h0cm.JPG')
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
detectPose(image, pose, display=True)