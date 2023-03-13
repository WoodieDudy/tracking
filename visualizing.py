import os.path
from typing import ClassVar

import numpy as np
from PIL import ImageFont, ImageDraw, Image


class TrackVisualizer:
    def __init__(self):
        self._border_size = 6
        self._border_color = (100, 178, 25)
        self._text_color = (45, 200, 45)
        self._clothes_violation_border_color = (255, 0, 0)
        self._clothes_default_border_color = (0, 255, 0)

    def draw_box(self, draw, coords, border_color=None):
        border_color = border_color or self._border_color

        x1, y1, x2, y2 = coords
        draw.rectangle((x1, y1, x2, y2), outline=border_color, width=self._border_size)

    def draw_label(self, draw, coords, label, font, down=True):
        x1, y1, x2, y2 = coords
        text_bg_w, text_bg_h = draw.textsize(label, font)
        if down:
            text_x1, text_y1 = x2, y2 - text_bg_h
        else:
            text_x1, text_y1 = x2, y1

        draw.rectangle((text_x1, text_y1, text_x1 + text_bg_w, text_y1 + text_bg_h), fill='black')
        draw.text((text_x1, text_y1), label, font=font, fill=self._text_color)

    def plot_tracking(self, image_rgb, bbs):
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        for i, bb in enumerate(bbs):
            self.draw_box(draw, bb.xyxy, border_color=(255, 0, 255))
            font = ImageFont.truetype("assets/arial.ttf", 20)
            self.draw_label(draw, bb.xyxy, bb.label, font, down=False)

        numpy_image = np.array(pil_image)  # type: ignore
        return numpy_image
