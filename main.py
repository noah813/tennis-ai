import argparse
from utils.Video import read_video, save_video
from tracker.TennisBall import TennisBall
from anaylysis.TennisBallSpeed import createReferenceLine, detect_ball_speed

def main(video_path, model_path):
    # Read video
    frames, fps = read_video(video_path)

    # Initialize TennisBall object
    tennis_ball = TennisBall(model_path)
    
    # Detect tennis ball in each frame
    ball_detections = tennis_ball.detect_frames(frames)

    # Draw reference line
    frames = createReferenceLine(frames)

    frames = detect_ball_speed(frames, ball_detections)

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

