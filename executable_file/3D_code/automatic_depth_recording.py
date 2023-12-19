import cv2
import os
import pykinect_azure as pykinect
import datetime
import mediapipe as mp



def main():
    num = 1
    current_datetime = datetime.datetime.now()
    formatted_time_for_filename = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    output_folder = f"depth_image_data/output_data_{formatted_time_for_filename}"
    os.makedirs(output_folder, exist_ok=True)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(2)

    recording = False

    while True:
        success, image = cap.read()



        pose = mp_pose.Pose(
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1)
        results = pose.process(image)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)

        # 繪製行人框
        # for (x, y, w, h) in boxes:
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if results.pose_landmarks:
            if not recording:
                pykinect.initialize_libraries(track_body=True)

                device_config = pykinect.default_configuration
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
                device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
                device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

                video_filename = os.path.join(output_folder, f"output_{num}.mkv")
                recording = True

                device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

            elif recording:
                capture = device.update()

        else:
            if recording:
                recording = False
                device.close()
                device = None
                num =+1


        cv2.imshow('image', image)

        if cv2.waitKey(1) == 27 :
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


