import dataclasses

import torch


@dataclasses.dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    conf: float
    cls_id: int = None

    @property
    def xyxy(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def to_tensor(self):
        # no cls yet?
        return torch.tensor([self.x_min, self.y_min, self.x_max, self.y_max, self.conf])
