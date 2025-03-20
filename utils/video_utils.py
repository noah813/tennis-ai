import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fames.append(frame)
    cap.release()
    return fames
def save_video(output_video_frames, output_video_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved at {output_video_path}")