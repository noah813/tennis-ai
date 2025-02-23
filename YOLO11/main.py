from utils import read_video, save_video
import TennisBall
import argparse

def main(video_path, model_path):

    frames, fps = read_video(video_path)

    tennis_ball = TennisBall(model_path)
    
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