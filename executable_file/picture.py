from views import *
import cv2

image = cv2.imread('img/wei_c300cm_h30cm.jpg')
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
detectPose(image, pose, display=True)


