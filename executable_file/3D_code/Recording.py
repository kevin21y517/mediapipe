import cv2
import os
import pykinect_azure as pykinect

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# video_folder = "video"
# if not os.path.exists(video_folder):
#     os.makedirs(video_folder)

# # 设置视频编码器
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # 创建一个名字基于当前时间的唯一文件名
# output_filename = os.path.join(video_folder, f"out_video.avi")
# out = cv2.VideoWriter(output_filename, fourcc, 30.0, (1920, 1080))  #偵率

# while True:
#     success, image = cap.read()
#     cv2.imshow('MediaPipe Pose', image)
#     out.write(image)
#     if not success:
#         print("Failed to write image to video.")

#     if cv2.waitKey(1) == 27:
#         break
# out.release()


#!/usr/bin/env python
#! --*-- coding:utf-8 --*--
import cv2
import time

import cv2
import time

if __name__ == '__main__':
    # 启动默认相机
    video = cv2.VideoCapture(0)

    # 获取 OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # 对于 webcam 不能采用 get(CV_CAP_PROP_FPS) 方法
    # 而是：
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS): {0}".format(fps))

    # Number of frames to capture
    num_frames = 200
    print("Capturing {0} frames".format(num_frames))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = 'output_video.avi'
    out = cv2.VideoWriter(output_filename, fourcc, fps, (int(video.get(3)), int(video.get(4))))

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = video.read()
        cv2.imshow('Frame', frame)  # Display the current frame
        out.write(frame)  # Write the frame to the video file
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken: {0} seconds".format(seconds))

    # 计算FPS，alculate frames per second
    fps = num_frames / seconds
    print("Estimated frames per second: {0}".format(fps))

    # 释放 video
    video.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()  # Close the OpenCV window
