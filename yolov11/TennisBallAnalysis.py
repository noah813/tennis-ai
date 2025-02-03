from detection.TennisBall import TennisBall
import cv2
import argparse
import sys
import os

# Import general utility functions
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.Video import read_video, save_video

def main(video_path, model_path):
    # Read video
    frames = read_video(video_path)

    # Initialize TennisBall object
    tennis_ball = TennisBall(model_path)
    
    # Detect tennis ball in each frame
    ball_detections = tennis_ball.detect_frames(frames)
    outputVideoFrames = tennis_ball.draw_bboxes(frames, ball_detections)

    save_video(outputVideoFrames)

if __name__ == "__main__":
    # Arguments list
    parser = argparse.ArgumentParser(description='Analyze video using trained model and save the output.')
    parser.add_argument('--modelPath', type=str, required=True, help='Path to the model file')
    parser.add_argument('--videoPath', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    
    main(args.videoPath, args.modelPath)

