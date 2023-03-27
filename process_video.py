import cv2
from tqdm import tqdm

from detectors import Yolov8Detector
from trackers import ByteTracker
from cv2wrappers import VideoCapture
from visualizing import TrackVisualizer


video_path = '/home/gk/projects/nir_tracking/data/videos/basket.mp4'

cap = VideoCapture(video_path).rgb()

detector = Yolov8Detector()
tracker = ByteTracker(fps=30, video_shape=cap.size)
video_writer = cv2.VideoWriter(
    'res.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    cap.size
)
visualizer = TrackVisualizer()

for i, frames_batch in enumerate(tqdm(cap.iterate_batched(64))):
    batch_preds = detector.inference(frames_batch)
    tracking_objects_batch = tracker.update(batch_preds)

    for frame, tracking_objects, bbs in zip(frames_batch, tracking_objects_batch, batch_preds):
        # plotted_frame = visualizer.plot_tracking(frame, bbs, box_color=(0, 255, 0))
        visualizer.plot_tracking(frame, tracking_objects)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if i == 20:
        break
video_writer.release()
