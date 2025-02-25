import cv2
import pickle
import os
import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.video_visualizer import VideoVisualizer

video = cv2.VideoCapture('input.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

video_writer = cv2.VideoWriter('out.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

cfg = get_cfg()
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
predictor = DefaultPredictor(cfg)

v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

def runOnVideo(video, maxFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """
    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        outputs = predictor(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        yield visualization
        readFrames += 1
        if readFrames > maxFrames:
            break

for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):
    video_writer.write(visualization)

video.release()
video_writer.release()
cv2.destroyAllWindows()

