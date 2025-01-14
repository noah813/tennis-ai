import cv2
import argparse
import torch
from ultralytics import YOLO  # 假設 YOLOv11 模型在 yolov11 模塊中
import os

# 設置命令行參數解析
parser = argparse.ArgumentParser(description='Detect tennis balls in a video using a trained YOLO model.')
parser.add_argument('--videoPath', type=str, required=True, help='Path to the video file')
parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained YOLO model')
args = parser.parse_args()  # 修正 parseArgs 為 parse_args

# 加載訓練好的模型
model = YOLO(args.modelPath)
model.to('cpu')  # 強制使用 CPU

# 打開影片檔案
videoPath = args.videoPath
cap = cv2.VideoCapture(videoPath)

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error: 無法打開影片檔案")
    exit()

# 獲取影片的幀率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 創建 VideoWriter 對象
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型進行檢測
    results = model.predict(frame, device='cpu')  # 使用 CPU 進行檢測

    # 繪製檢測結果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 假設 boxes 屬性包含邊界框
        scores = result.boxes.conf.cpu().numpy()  # 假設 conf 屬性包含置信度
        classes = result.boxes.cls.cpu().numpy()  # 假設 cls 屬性包含類別索引

        for box, conf, cls in zip(boxes, scores, classes):
            label = f'{model.names[int(cls)]} {conf:.2f}'  # 假設 YOLO 模型有 names 屬性
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 寫入幀到影片
    out.write(frame)

# 釋放影片對象
cap.release()
out.release()
# cv2.destroyAllWindows()
