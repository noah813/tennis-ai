import cv2

# 打開影片檔案
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error: 無法打開影片檔案")
    exit()

import cv2
import os

# 打開影片檔案
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error: 無法打開影片檔案")
    exit()

# 創建保存圖片的目錄
output_dir = 'frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 儲存每一幀的圖像
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

    print(f"成功保存 {frame_filename}")

# 釋放影片對象
cap.release()
print("所有幀已保存")