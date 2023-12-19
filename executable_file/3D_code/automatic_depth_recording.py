import cv2
import os
import pykinect_azure as pykinect
import datetime
import mediapipe as mp



def main():
    current_datetime = datetime.datetime.now()
    formatted_time_for_filename = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    output_folder = f"depth_image_data/output_data_{formatted_time_for_filename}"
    os.makedirs(output_folder, exist_ok=True)

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(1)

    recording = False

    while True:
        success, image = cap.read()
        cv2.imshow('image', image)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pose = mp_pose.Pose(
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1)
        results = pose.process(image)


        if results.pose_landmarks:
            if not recording:
                pykinect.initialize_libraries(track_body=True)

                device_config = pykinect.default_configuration
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
                device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
                device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

                num = 1
                video_filename = os.path.join(output_folder, f"output_{num}.mkv")
                recording = True

                device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

            elif recording:
                capture = device.update()
                ret_color, color_image = capture.get_transformed_color_image()
                ret_depth, depth_color_image = capture.get_colored_depth_image()

        elif results.pose_landmarks is None:
            if recording:
                recording = False
                device.close()
                device = None
                num = num+1

        if cv2.waitKey(1) == 27 :
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


