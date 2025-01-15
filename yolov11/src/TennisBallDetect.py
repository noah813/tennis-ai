from ultralytics import YOLO

# 設置命令行參數解析
parser = argparse.ArgumentParser(description='Detect tennis balls in a video using a trained YOLO model.')
parser.add_argument('--videoPath', type=str, required=True, help='Path to the video file')
parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained YOLO model')
args = parser.parse_args()  # 修正 parseArgs 為 parse_args

# 加載訓練好的模型
model = YOLO(args.modelPath)

# 載入視頻 
videoPath = args.videoPath

model(videoPath, show = False, save = True)
