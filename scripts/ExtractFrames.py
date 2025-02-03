import cv2
import os
import sys
import argparse

def extract_frames(video_path, frame_interval, output_base_dir='../images/clip'):
    if not os.path.exists(video_path):
        print(f"Error: 影片路徑 {video_path} 不存在。")
        sys.exit(1)

    if frame_interval < 1:
        print("Error: frameInterval 必須大於 0。")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 無法打開影片檔案")
        sys.exit(1)

    # 根據 output_base_dir 建立一個不重複的資料夾
    dir_count = 1
    output_dir = os.path.join(output_base_dir + str(dir_count))
    while os.path.exists(output_dir):
        dir_count += 1
        output_dir = os.path.join(output_base_dir + str(dir_count))
    os.makedirs(output_dir, exist_ok=True)
    print(f"圖片將存放在 {output_dir}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"成功保存 {frame_filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print("所有幀已保存")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video.')
    parser.add_argument('--videoPath', type=str, required=True, help='Path to the video file')
    parser.add_argument('--frameInterval', type=int, required=True, help='Interval of frames to save')
    args = parser.parse_args()

    extract_frames(args.videoPath, args.frameInterval)