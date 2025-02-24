import cv2 
import sys 

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print(f"Error: Can't open video {video_path}")
        sys.exit(1)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames, fps

def save_video(frame_array, output_path, fps):
    height, width, _ = frame_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frame_array:
        out.write(frame)
    out.release()