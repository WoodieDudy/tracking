import math
from typing import Union, Any, TypeVar, Generic

import cv2

T = TypeVar("T")


class VideoPreprocessor(Generic[T]):
    def __init__(self):
        self._resize = None
        self._interpolation = None
        self._convert_to = None

    def resized(self, size: int | tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> T:
        self._resize = size if isinstance(size, tuple) else (size, size)
        self._interpolation = interpolation
        return self

    def convert_to(self, color: int | None) -> T:
        self._convert_to = color
        return self

    def rgb(self) -> T:
        return self.convert_to(cv2.COLOR_BGR2RGB)

    def bgr(self) -> T:
        return self.convert_to(None)

    def gray(self) -> T:
        return self.convert_to(cv2.COLOR_BGR2GRAY)

    @property
    def color_conversion(self) -> int | None:
        return self._convert_to

    def _process(self, frame: Any) -> Any:
        if self._resize:
            frame = cv2.resize(frame, self._resize, interpolation=self._interpolation)
        if self._convert_to:
            frame = cv2.cvtColor(frame, self._convert_to)
        return frame


class VideoCapture(VideoPreprocessor['VideoCapture']):
    def __init__(self, video_src: Union[str, cv2.VideoCapture]):
        super().__init__()

        if isinstance(video_src, str):
            self._cap = cv2.VideoCapture(video_src)
        else:
            self._cap = video_src

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frames_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_frame = 0

    def set_current_frame(self, frame: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self._current_frame = frame

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    @property
    def length_in_seconds(self) -> float:
        return self._frames_count / self._fps

    @property
    def capture(self) -> cv2.VideoCapture:
        return self._cap

    @property
    def fps(self) -> int:
        return int(self._fps)

    @property
    def precise_fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height) if self._resize is None else self._resize

    @property
    def frames_count(self) -> int:
        return self._frames_count

    def get_frame(self, idx: int) -> Any:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None

        return self._process(frame)

    def __del__(self):
        self._cap.release()

    def __iter__(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
        return self

    def __next__(self):
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        return self._process(frame)

    def __len__(self):
        return self._frames_count

    def __getitem__(self, idx):
        return self.get_frame(idx)

    def __repr__(self):
        return f"VideoCapture: {self._frames_count} frames, {self._fps} fps," \
               f" {self._width}x{self._height}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cap.release()

    def __bool__(self):
        return self._cap.isOpened()

    def iterate_batched(self, batch_size: int, end_frame: int | None = None) -> 'BatchIterator':
        return BatchIterator(self, batch_size, end_frame)

    def iterate_batched_with_step(self, batch_size: int, step: int,
                                  end_frame: int | None = None) -> 'BatchStepIterator':
        return BatchStepIterator(self, batch_size, step, end_frame)


class BatchIterator:
    def __init__(self, video_capture: 'VideoCapture', batch_size: int, end_frame: int | None):
        self._capture = video_capture
        self._batch_size = batch_size
        self._end_frame = self._capture.frames_count
        if end_frame is not None:
            self._end_frame = min(self._end_frame, end_frame)

        self._start = self._current_frame = self._capture.current_frame
        self._iterator = iter(self._capture)

    def __next__(self):
        if self._current_frame >= self._end_frame:
            raise StopIteration

        frames = []

        count = self._batch_size
        if self._current_frame + self._batch_size > self._end_frame:
            count = self._end_frame - self._current_frame

        for _ in range(count):
            frame = next(self._iterator)
            frames.append(frame)

        self._current_frame += count
        return frames

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return math.ceil((self._end_frame - self._start) / self._batch_size)


class BatchStepIterator:
    def __init__(self, capture: 'VideoCapture', batch_size: int, step: int, end_frame: int | None):
        if step < 1:
            raise ValueError("Step must be greater than 0")

        self._capture = capture
        self._batch_size = batch_size
        self._step = step
        self._end_frame = self._capture.frames_count
        if end_frame is not None:
            self._end_frame = min(self._end_frame, end_frame)

        self._start = self._current_frame = self._capture.current_frame

    def __next__(self) -> list[tuple[int, Any]]:
        if self._current_frame >= self._end_frame:
            raise StopIteration

        count = self._batch_size
        if self._current_frame + self._batch_size * self._step > self._end_frame:
            count = (self._end_frame - self._current_frame) // self._step

        if count == 0:
            raise StopIteration

        frames = []

        for _ in range(count):
            frame = self._capture.get_frame(self._current_frame)
            if frame is None:
                break

            frames.append((self._current_frame, frame))
            self._current_frame += self._step

        if not frames:
            raise StopIteration

        return frames

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return math.ceil((self._end_frame - self._start) / (self._batch_size * self._step))
