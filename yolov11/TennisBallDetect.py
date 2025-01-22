import argparse 
from ultralytics import YOLO
from utils import (read_video, save_video)
from trakers.BallTraker import BallTracker

# 設置命令行參數解析
parser = argparse.ArgumentParser(description='Detect tennis balls in a video using a trained YOLO model.')
parser.add_argument('--videoPath', type=str, required=True, help='Path to the video file')
parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained YOLO model')
args = parser.parse_args()  # 修正 parseArgs 為 parse_args

# 載入視頻 
videoPath = args.videoPath
frames = read_video(videoPath)

# 初始化 BallTracker
ball_tracker = BallTracker(args.modelPath)

# 檢測網球
ball_detections = ball_tracker.detect_frames(frames)

# 繪製邊界框
output_frames = ball_tracker.draw_bboxes(frames, ball_detections)

# 保存視頻
save_video(output_frames, 'output_video.avi')
