import cv2
import os
import pykinect_azure as pykinect
import datetime
import time
import numpy as np


class Recording():
    def __init__(self):
        self.frame_count = None
        self.elapsed_time = None
        self.should_exit = False
        self.ID = 1
        self.current_datetime = datetime.datetime.now()
        self.formatted_time_for_filename = self.current_datetime.strftime("%Y-%m-%d_%H%M%S")
        self.output_folder = f"depth_image_data/output_data_{self.formatted_time_for_filename}"

    def initialize_libraries(self):
        pykinect.initialize_libraries(track_body=True)

    def get_user_input(self):
        window_name = 'ID'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 400)

        text = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 200)
        font_scale = 1
        font_color = (255, 255, 255)
        line_thickness = 2

        frame_rate = self.frame_count / self.elapsed_time
        elapsed_time_str = f"duration: {self.elapsed_time} second"
        frame_rate_str = f"Frame rate: {frame_rate:.2f} fps"

        while True:
            image = np.zeros((400, 600, 3), dtype=np.uint8)  # Create a black image
            cv2.putText(image, elapsed_time_str , (10, 30), font, 0.7, font_color, 1, cv2.LINE_AA)
            cv2.putText(image, frame_rate_str , (10, 60), font, 0.7, font_color, 1, cv2.LINE_AA)
            cv2.putText(image, text, position, font, font_scale, font_color, line_thickness, cv2.LINE_AA)
            cv2.imshow(window_name, image)

            key = cv2.waitKey(20)

            if key == 13:  # Enter key
                self.ID = text
                break
            elif key == 27:
                self.should_exit = True
                break
            elif key == 8:  # Backspace key
                text = text[:-1]
            elif key != -1:
                text += chr(key)

        cv2.destroyWindow(window_name)

    def main(self):
        self.initialize_libraries()

        os.makedirs(self.output_folder, exist_ok=True)

        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

        video_filename = os.path.join(self.output_folder, f"{self.ID}_output.mkv")
        device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)
        cv2.namedWindow('Depth Image with Values', cv2.WINDOW_NORMAL)

        start_time = time.time()
        self.frame_count = 0

        while True:
            capture = device.update()

            ret_color, color_image = capture.get_transformed_color_image()
            ret_depth, depth_color_image = capture.get_colored_depth_image()

            if ret_color and ret_depth and color_image is not None and depth_color_image is not None:
                combined_image = cv2.addWeighted(color_image[:, :, :3], 0.9, depth_color_image, 0.1, 0)
                cv2.putText(combined_image, f'ID : {self.ID}',(20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow('Depth Image with Values', combined_image)

            self.frame_count += 1

            if cv2.waitKey(1) == 32:
                end_time = time.time()
                self.elapsed_time = end_time - start_time
                self.get_user_input()
                device.close()
                break
            if cv2.waitKey(1) == 27:
                self.should_exit = True
                device.close()
                break



    def run(self):
        cv2.namedWindow('Depth Image with Values', cv2.WINDOW_NORMAL)
        while True:
            key = cv2.waitKey(1)
            if key == 27 or self.should_exit:
                cv2.destroyAllWindows()
                break
            elif key == 32:
                self.main()

if __name__ == "__main__":
    record = Recording()
    record.run()
