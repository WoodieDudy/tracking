import cv2
from tqdm import tqdm

from detectors import Yolov8Detector
from trackers import ByteTracker
from cv2wrappers import VideoCapture
from visualizing import TrackVisualizer


video_path = '/home/gk/projects/nir_tracking/data/videos/basket.mp4'
cap = VideoCapture(video_path).rgb()

detector = Yolov8Detector()
tracker = ByteTracker(fps=cap.fps, video_shape=cap.size)
video_writer = cv2.VideoWriter(
    'res.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.fps,
    cap.size
)
visualizer = TrackVisualizer()

for frames_batch in tqdm(cap.iterate_batched(64)):
    batch_preds = detector.inference(frames_batch)
    tracking_objects_batch = tracker.update(batch_preds)

    for frame, tracking_objects in zip(frames_batch, tracking_objects_batch):
        plotted_frame = visualizer.plot_tracking(frame, tracking_objects)
        video_writer.write(cv2.cvtColor(plotted_frame, cv2.COLOR_RGB2BGR))

video_writer.release()
