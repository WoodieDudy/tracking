import cv2
from tqdm import tqdm

from detectors import Yolov8Detector
from trackers import ByteTracker, DeepSortReid
from cv2wrappers import VideoCapture
from visualizing import TrackVisualizer


video_path = '/home/gk/projects/nir_tracking/data/videos/basket2.mp4'

cap = VideoCapture(video_path).rgb()

detector = Yolov8Detector()
# tracker = ByteTracker(fps=cap.fps, video_shape=cap.size)
tracker = DeepSortReid()
video_writer = cv2.VideoWriter(
    'basket2_deep_sort.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.fps,
    cap.size
)
visualizer = TrackVisualizer()

for i, frames_batch in enumerate(tqdm(cap.iterate_batched(64))):
    batch_preds = detector.inference(frames_batch)
    tracking_objects_batch = tracker.update(frames_batch, batch_preds)

    for frame, tracking_objects, bbs in zip(frames_batch, tracking_objects_batch, batch_preds):
        # plotted_frame = visualizer.plot_tracking(frame, bbs, box_color=(0, 255, 0))
        visualizer.plot_tracking(frame, tracking_objects)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video_writer.release()
