import cv2
import os
import pykinect_azure as pykinect
import datetime
import time
import numpy as np


class Recording():
    def __init__(self):
        self.current_datetime = datetime.datetime.now()
        self.formatted_time_for_filename = self.current_datetime.strftime("%Y-%m-%d_%H%M%S")
        self.output_folder = f"depth_image_data/output_data_{self.formatted_time_for_filename}"
        self.should_exit = False
        self.ID = 1
        self.run_time = 60

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

        while True:
            # image = self.create_image(text, font, position, font_scale, font_color, line_thickness)
            image = np.zeros((400, 600, 3), dtype=np.uint8)  # Create a black image
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
        # Initialize the library, if the library is not found, add the library path as an argument
        pykinect.initialize_libraries(track_body=True)

        self.start_time = time.time()
        os.makedirs(self.output_folder, exist_ok=True)

        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

        # Start device
        video_filename = os.path.join(self.output_folder, f"{self.ID}_output.mkv")
        device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

        # cap = cv2.VideoCapture(2)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # output_2D_filename = os.path.join(self.output_folder, f"{self.ID}_2D_out_video.avi")
        # video_2D = cv2.VideoWriter(output_2D_filename, fourcc, 5.0, (640, 480))  #偵率

        while True:
            # Get capture
            capture = device.update()

            ret_color, color_image = capture.get_transformed_color_image()
            ret_depth, depth_color_image = capture.get_colored_depth_image()
            # ret, depth_image = capture.get_colored_depth_image()

            if ret_color and ret_depth and color_image is not None and depth_color_image is not None:
                combined_image = cv2.addWeighted(color_image[:, :, :3], 0.9, depth_color_image, 0.1, 0)
                cv2.putText(combined_image, f'ID : {self.ID}',(20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow('Depth Image with Values', combined_image)

            # success, image = cap.read()
            # video_2D.write(image)
            # cv2.imshow('2D Image', image)

            # Press Enter key to stop
            if cv2.waitKey(100) ==32 or (self.start_time > 100 and time.time()
                    - self.start_time >= self.run_time):
                # video_2D.release()
                self.get_user_input()
                break
            if cv2.waitKey(50) ==27:
                self.should_exit = True
                break



    def run(self):
        cv2.namedWindow('Depth Image with Values', cv2.WINDOW_NORMAL)
        while True:
            key = cv2.waitKey(30)
            if key == 27 or self.should_exit:  # 如果按下 ESC，跳出迴圈
                cv2.destroyAllWindows()
                break
            elif key == 32:  # 如果按下空白鍵
                # self.num += 1
                self.main()  # 執行你的主程式


if __name__ == "__main__":
    record = Recording()
    record.run()