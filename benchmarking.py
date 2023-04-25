from pathlib import Path
import os

from benchmarks.becnhmark_runner import BenchmarkRunner
from benchmarks.impl.mota_metrics import MOTAMetrics
from benchmarks.impl.track_writer import TrackWriter
from benchmarks.tracking_benchmark import TrackingBenchmark
from detectors import Yolov8Detector, DetectionsLoader
from trackers import ByteTracker, DeepSort
from trackers.deep_sort.tracker import RealDeepSort

dataset_path = Path('train')
batch_size = 64

videos_path = Path('videos')
os.makedirs(str(videos_path), exist_ok=True)
byte_track_benchmark = TrackingBenchmark(
    # detector=DetectionsLoader(Path(dataset_path), 'det'),
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=dataset_path,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

deep_sort_benchmark = TrackingBenchmark(
    # detector=DetectionsLoader(Path(dataset_path), 'det'),
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: RealDeepSort(),
    dataset_path=dataset_path,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSort_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

runner = BenchmarkRunner([
    # byte_track_benchmark,
    deep_sort_benchmark,
])

runner.run()
