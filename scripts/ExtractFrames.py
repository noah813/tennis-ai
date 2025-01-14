import cv2
import os
import argparse

# 設置命令行參數解析
parser = argparse.ArgumentParser(description='Process video to extract frames.')
parser.add_argument('--videoPath', type=str, required=True, help='Path to the video file')
parser.add_argument('--frameInterval', type=int, required=True, help='Interval of frames to save')
args = parser.parse_args()

# 打開影片檔案
videoPath = args.videoPath
cap = cv2.VideoCapture(videoPath)

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error: 無法打開影片檔案")
    exit()

# 創建保存圖片的目錄
output_base_dir = '../images/clip'
output_dir = output_base_dir + '1'
dir_count = 1

while os.path.exists(output_dir):
    dir_count += 1
    output_dir = output_base_dir + str(dir_count)

os.makedirs(output_dir)

frame_count = 0
frame_interval = args.frameInterval  # 每多少幀截一次

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # 儲存每一幀的圖像
        frame_filename = os.path.join(output_dir, f'frame_{frame_count // frame_interval:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"成功保存 {frame_filename}")

    frame_count += 1

# 釋放影片對象
cap.release()
print("所有幀已保存")