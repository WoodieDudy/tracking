from pathlib import Path

from benchmarks.becnhmark_runner import BenchmarkRunner
from benchmarks.impl.mota_metrics import MOTAMetrics
from benchmarks.impl.track_writer import TrackWriter
from benchmarks.tracking_benchmark import TrackingBenchmark
from detectors import Yolov8Detector
from trackers import ByteTracker, DeepSort

dataset_path = 'mot20/train'
batch_size = 64

videos_path = Path('videos')

byte_track_benchmark = TrackingBenchmark(
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=dataset_path,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence,
                                                                videos_path / f'ByteTrack_{root_dir[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

deep_sort_benchmark = TrackingBenchmark(
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: DeepSort(),
    dataset_path=dataset_path,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence,
                                                                videos_path / f'DeepSort_{root_dir[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

runner = BenchmarkRunner([
    byte_track_benchmark,
    deep_sort_benchmark
])

runner.run()
