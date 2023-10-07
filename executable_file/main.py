from views import *
import cv2   # pip install opencv-python  

cap = cv2.VideoCapture(0)  # 打開默認攝像頭

# 檢查攝像頭是否打開
if not cap.isOpened():
        print("無法打開攝像頭，請檢查是否已連接攝像頭。")
else:
    while True:
        success, image = cap.read() # 如果成功讀取攝像頭的一幀影象
        if not success: 
            print("無法讀取攝像頭影象。")
            break
        cv2.imshow('Camera', image) # 顯示攝像頭影象，窗口名稱為Camera
        # 如果按下空格键，则执行姿势估计

        if cv2.waitKey(1) == 32:  # 空格键的ASCII码是32
            run_time=300
            cv2.destroyWindow('Camera') # 關閉攝像頭視窗
            process_pose_estimation(cap, run_time)  #估計姿勢，傳遞參數
            if cv2.waitKey(1) == 27: # 按下ESC键退出
                break
    
    # 釋放資源，關閉攝像頭和視窗
    cap.release()
    cv2.destroyAllWindows()