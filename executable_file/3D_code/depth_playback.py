import cv2
import time
import pykinect_azure as pykinect

if __name__ == "__main__":
    file = "output_data_2023-12-08_234567"
    video_filename = f"depth_image_data/{file}/output.mkv"

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Start playback
    playback = pykinect.start_playback(video_filename)

    playback_config = playback.get_record_configuration()
    # print(playback_config)

    cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)

    delay = 0.145  # 延遲時間（秒）
    while True:

        # Get camera capture
        ret, capture = playback.update()

        if not ret:
            break

        # Get color image
        ret_color, color_image = capture.get_transformed_color_image()

        # Get the colored depth
        ret_depth, depth_color_image = capture.get_colored_depth_image()

        if not ret_color or not ret_depth:
            continue

        # Plot the image
        combined_image = cv2.addWeighted(color_image[:, :, :3], 1.0, depth_color_image, 0.0, 0)
        cv2.imshow('Depth Image', combined_image)

        time.sleep(delay)

        # Press q key to stop
        if cv2.waitKey(1) == 27:
            break
