import cv2
import os
import json
import pykinect_azure as pykinect
import datetime

def main():
    # Initialize the library, if the library is not found, add the library path as an argument
    pykinect.initialize_libraries(track_body=True)

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
    device = pykinect.start_device(config=device_config)

    # Start body tracker
    body_tracker = pykinect.start_body_tracker()

    while True:
        # Get capture
        capture = device.update()

        # Get the depth image
        ret_depth, depth_image = capture.get_depth_image()

        # Display depth image with depth values
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image with Values', depth_colormap)

        if ret_depth is not None:
            depth_image_filename = os.path.join(output_folder, f"transformed_color_image_{frame_count}.png")
            cv2.imwrite(depth_image_filename, depth_colormap)
            # Process and save depth information to JSON
            depth_data = {}
            for row in range(depth_image.shape[0]):
                for col in range(depth_image.shape[1]):
                    depth_value = int(depth_image[row, col].item())  # 轉換為 int
                    depth_data[f"pixel_{row}_{col}"] = depth_value

            json_filename = os.path.join(output_folder, f"depth_data_{frame_count}.json")
            with open(json_filename, 'w') as json_file:
                json.dump(depth_data, json_file, indent=4)

        frame_count += 1

        # Press q key to stop
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
