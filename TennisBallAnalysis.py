from utils.Video import read_video, save_video
import cv2
import argparse

def main(video_path):
    frames = read_video(video_path)
    
    save_video(frames)

if __name__ == "__main__":
    # Arguments list
    parser = argparse.ArgumentParser(description='Analyze video using trained model and save the output.')
    parser.add_argument('--videoPath', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    
    main(args.videoPath)

