import cv2
import os
import json
import pykinect_azure as pykinect

if __name__ == "__main__":
    file = "output_data_2023-12-08_234567"
    video_filename = f"depth_image_data/{file}/output.mkv"
    output_folder = f"depth_image_data/{file}"
    frame_count = 0

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Start playback
    playback = pykinect.start_playback(video_filename)

    playback_config = playback.get_record_configuration()
    # print(playback_config)

    cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
    while True:

        # Get camera capture
        ret, capture = playback.update()

        if not ret:
            break

        # Get color image
        ret_color, color_image = capture.get_transformed_color_image()

        # Get the colored depth
        ret_depth, depth_color_image = capture.get_colored_depth_image()

# Plot the image


        if ret_depth is not None:
            depth_colormap = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_color_image, 0.3, 0)
            cv2.imshow('Depth Image', depth_colormap)

            depth_image_filename = os.path.join(output_folder, f"transformed_color_image_{frame_count}.png")
            cv2.imwrite(depth_image_filename, depth_colormap)
            # Process and save depth information to JSON
            depth_data = {}
            for row in range(depth_color_image.shape[0]):
                for col in range(depth_color_image.shape[1]):
                    depth_value = depth_color_image[row, col][0]  # Access the correct element
                    depth_data[f"pixel_{row}_{col}"] = int(depth_value)

            json_filename = os.path.join(output_folder, f"depth_data_{frame_count}.json")
            with open(json_filename, 'w') as json_file:
                json.dump(depth_data, json_file, indent=4)

        frame_count += 1




        # Press q key to stop
        if cv2.waitKey(1) == 27:
            break
