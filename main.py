from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker
from trackers import BallTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    input_video_path = "assets/videos/input2.mp4"
    video_frames = read_video(input_video_path)

    # Detect the video
    ## Detect the players in the video
    player_tracker = PlayerTracker(model_path="assets/models/yolo11x.pt")
    player_detection = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="assets/stubs/player_detection.pkl")
    ## Detect the ball in the video
    ball_tracker = BallTracker(model_path="assets/models/tennis_ball.pt")
    ball_detection = ball_tracker.detect_frames(video_frames,
                                                read_from_stub=True,
                                                stub_path="assets/stubs/ball_detection.pkl")
    ball_detection = ball_tracker.interpolate_ball_position(ball_detection)
    ## Detect the court keypoints in the video
    court_line_detector = CourtLineDetector(model_path="assets/models/keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])
    ## Choose players that are inside the court
    player_detection = player_tracker.choose_and_filter_players(court_keypoints, player_detection)
    
    # Draw output
    ## Draw bounding boxes around the players
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    ## Draw bounding boxes around the ball
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detection)
    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_videos(output_video_frames, court_keypoints)
    ## Draw frame number on the top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    save_video(video_frames, "assets/videos/output.mp4")
if __name__ == "__main__":
    main()