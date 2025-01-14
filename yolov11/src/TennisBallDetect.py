import cv2
import argparse
import torch

# 設置命令行參數解析
parser = argparse.ArgumentParser(description='Detect tennis balls in a video using a trained YOLO model.')
parser.add_argument('--videoPath', type=str, required=True, help='Path to the video file')
parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained YOLO model')
args = parser.parse_args()

# 加載訓練好的模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.modelPath)

# 打開影片檔案
videoPath = args.videoPath
cap = cv2.VideoCapture(videoPath)

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error: 無法打開影片檔案")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型進行檢測
    results = model(frame)

    # 繪製檢測結果
    for *box, conf, cls in results.xyxy[0]:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow('Tennis Ball Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放影片對象
cap.release()
cv2.destroyAllWindows()
