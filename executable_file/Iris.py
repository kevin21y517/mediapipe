import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# 初始化摄像头
cap = cv2.VideoCapture(0)

fliph_img = True

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            if fliph_img is True:
                image = cv2.flip(image, 1)

            # To improve performance, mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_h, frame_w, _ = image.shape
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    iris_boundary = mp_face_mesh.FACEMESH_IRISES
                    for connection in iris_boundary:
                        for point in connection:
                            x, y = int(face_landmarks.landmark[point].x * frame_w), int(face_landmarks.landmark[point].y * frame_h)
                            z = face_landmarks.landmark[point].z  # 获取 z 坐标
                            cv2.circle(image, (x, y), 1, (0, 255, 0), 2)

                            # 显示 (x, y, z) 坐标信息
                        # cv2.putText(image, f'({x}, {y}, {z:.2f})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('Eye Contours', image)
            k = cv2.waitKey(1)
            if k == 27:
                break

cap.release()
