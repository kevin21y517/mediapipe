import cv2
import pykinect_azure as pykinect
import time

if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as an argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

    # Start device
    video_filename = "output.mkv"
    device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

    cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)

    # Variables for frame rate calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Get capture
        capture = device.update()
        ret_color, color_image = capture.get_transformed_color_image()
        ret_depth, depth_color_image = capture.get_colored_depth_image()

        # Get the color depth image from the capture
        ret, depth_image = capture.get_colored_depth_image()

        # if not ret:
        #     continue

        # # Plot the image
        # cv2.imshow('Depth Image', depth_image)
        if ret_color and ret_depth and color_image is not None and depth_color_image is not None:
                combined_image = cv2.addWeighted(color_image[:, :, :3], 0.9, depth_color_image, 0.1, 0)
                cv2.imshow('Depth Image', combined_image)

        # Increment frame count
        frame_count += 1

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

    # Calculate frame rate
    end_time = time.time()
    elapsed_time = end_time - start_time
    frame_rate = frame_count / elapsed_time

    print(f"Frame rate: {frame_rate:.2f} fps")

    # Release the device
    device.close()
    cv2.destroyAllWindows()
