from views import *
import cv2  # pip install opencv-python

image = cv2.imread('C:/Venv/Project/Kevin_mediapipe/mediapipe/img/wei_c200cm_h10cm.jpg') # 讀取圖片(路徑)
#姿勢估計對象為pose，估計靜態圖片，最小的檢測信賴區間為0.3，模型複雜度為2
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2) 
detectPose(image, pose, display=True)   #後製圖片，顯示參數在圖片上