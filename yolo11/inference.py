from ball_tracker import BallTracker
from utils import read_video, save_video

def inference(model_path,video_path, output_path):
    ball_tracker = BallTracker(model_path)
    frames, fps = read_video(video_path)
    ball_detections = ball_tracker.detect_frames(frames)
    output_video_frames = draw_bboxes(frames, ball_detections)
    save_video(output_video_frames, output_path, fps)