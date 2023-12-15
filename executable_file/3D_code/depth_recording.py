import cv2
import os
import pykinect_azure as pykinect
import datetime
import time

def main():
    # Initialize the library, if the library is not found, add the library path as an argument
    pykinect.initialize_libraries(track_body=True)

    run_time = 20
    start_time = time.time()
    frame_count = 0
    current_datetime = datetime.datetime.now()
    formatted_time_for_filename = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    output_folder = f"depth_image_data/output_data_{formatted_time_for_filename}"
    os.makedirs(output_folder, exist_ok=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    # Start device
    video_filename = os.path.join(output_folder, "output_.mkv")
    device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

    # Start body tracker
    body_tracker = pykinect.start_body_tracker()

    while True:
        # Get capture
        capture = device.update()

        ret_color, color_image = capture.get_transformed_color_image()
        ret_depth, depth_color_image = capture.get_colored_depth_image()

        # Get the depth image
        ret_depth, depth_image = capture.get_depth_image()

        # Display depth image with depth values
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.9, depth_color_image, 0.1, 0)
        cv2.imshow('Depth Image with Values', combined_image)

        frame_count += 1

        # Press q key to stop
        if cv2.waitKey(1) == 27 or (start_time > 100 and time.time()
                - start_time >= run_time):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
