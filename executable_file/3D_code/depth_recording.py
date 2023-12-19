import cv2
import os
import pykinect_azure as pykinect
import datetime
import time

current_datetime = datetime.datetime.now()
formatted_time_for_filename = current_datetime.strftime("%Y-%m-%d_%H%M%S")
output_folder = f"depth_image_data/output_data_{formatted_time_for_filename}"
os.makedirs(output_folder, exist_ok=True)


def main(num):
    # Initialize the library, if the library is not found, add the library path as an argument
    pykinect.initialize_libraries(track_body=True)

    run_time = 60
    start_time = time.time()
    frame_count = 0

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    # Start device
    video_filename = os.path.join(output_folder, f"output_{num}.mkv")
    device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

    # Start body tracker
    body_tracker = pykinect.start_body_tracker()

    while True:
        # Get capture
        capture = device.update()

        ret_color, color_image = capture.get_transformed_color_image()
        ret_depth, depth_color_image = capture.get_colored_depth_image()

        if ret_color and ret_depth and color_image is not None and depth_color_image is not None:
            combined_image = cv2.addWeighted(color_image[:, :, :3], 0.9, depth_color_image, 0.1, 0)
        cv2.imshow('Depth Image with Values', combined_image)

        frame_count += 1

        # Press Enter key to stop
        if cv2.waitKey(1) ==13 or cv2.waitKey(1) ==27 or (start_time > 100 and time.time()
                - start_time >= run_time):
            break



if __name__ == "__main__":
    cv2.namedWindow('Depth Image with Values', cv2.WINDOW_NORMAL)
    num = 0
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # 如果按下 ESC，跳出迴圈
            cv2.destroyAllWindows()
            break
        elif key == 32:  # 如果按下空白鍵
            num += 1
            main(num)  # 執行你的主程式
