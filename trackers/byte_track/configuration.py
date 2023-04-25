import dataclasses


@dataclasses.dataclass
class BYTETrackerConfig:
    track_thresh: float
    track_buffer: int
    mot20: bool
    match_thresh: float
