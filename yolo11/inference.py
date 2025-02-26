import argparse
from utils import read_video, save_video
from .ball_tracker import BallTracker

def main(video_path, model_path):
    # Read video
    frames, fps = read_video(video_path)

    # Initialize TennisBall object
    tennis_ball = BallTracker(model_path)
    
    # Detect tennis ball in each frame
    ball_detections = tennis_ball.detect_frames(frames)

    # Draw output
    outputVideoFrames = tennis_ball.draw_bboxes(frames, ball_detections)

    save_video(outputVideoFrames)

if __name__ == "__main__":
    # Arguments list
    parser = argparse.ArgumentParser(description='Analyze video using trained model and save the output.')
    parser.add_argument('--modelPath', type=str, required=True, help='Path to the model file')
    parser.add_argument('--videoPath', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    
    main(args.videoPath, args.modelPath)