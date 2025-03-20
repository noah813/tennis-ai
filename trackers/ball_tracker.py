from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        id_name_dict = results.names
        ball_dict = {}
        for box in results.boxes:
            results = box.xyxy.tolist()[0]
            ball_dict[1] = results
        return ball_dict
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detection = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detection = pickle.load(f)
            return ball_detection
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detection.append(ball_dict)
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detection, f)
        return ball_detection
    
    def draw_bboxes(self, video_frame, ball_detection):
        output_video_frames = []
        for frame, ball_dict in zip(video_frame, ball_detection):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"ball", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
    
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()] # 1 is the track id, x is bboxes
        return ball_positions