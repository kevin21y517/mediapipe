import cv2
import os
import pykinect_azure as pykinect

cap = cv2.VideoCapture(2)
video_folder = "video"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# 设置视频编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 创建一个名字基于当前时间的唯一文件名
output_filename = os.path.join(video_folder, f"out_video.avi")
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))  #偵率

while True:
    success, image = cap.read()
    cv2.imshow('MediaPipe Pose', image)
    out.write(image)
    if not success:
        print("Failed to write image to video.")

    if cv2.waitKey(1) == 27:
        break
out.release()