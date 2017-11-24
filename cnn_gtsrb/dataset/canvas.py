"""
Canvas class used to paste images.
"""

import numpy as np
from PIL import Image, ImageDraw

BORDER_COLORS = [
    (255, 255, 255),
    (255, 0, 0),        # 1 = red
    (0, 255, 0),        # 2 = green
    (0, 0, 255),        # 3 = blue
    (255, 255, 0),      # 4 = yellow
    (0, 255, 255),      # 5 = cyan
    (255, 0, 255),      # 6 = purple
]

class Canvas():

    def __init__(self, image_size, w_count, h_count, gap_size=8, mode='L'):
        """
        :param image_size: The image size in pixel.
        :param w_count: Image count horizontally
        :param h_count: Image count vertically
        :param gap_size: The gap size in pixel
        """
        self.image_size = image_size
        self.w_count = w_count
        self.h_count = h_count
        self.mode = mode
        self.gap_size = gap_size
        self.canvas_w = self._pixel(w_count)
        self.canvas_h = self._pixel(h_count)

        self.canvas = Image.new('RGB', (self.canvas_w, self.canvas_h), color=(255, 255, 255))

    def _pixel(self, p):
        return (self.image_size + self.gap_size) * p + self.gap_size

    def paste(self, im, xpos, ypos, **kwargs):
        """Paste im to (xpos, ypos) position."""
        # If border=color, then draw a black border
        if kwargs.get('border'):
            border = kwargs.get('border')
            self.draw_border(xpos, ypos, BORDER_COLORS[border])

        x = self._pixel(xpos)
        y = self._pixel(ypos)

        self.canvas.paste(im, (x, y))

    def draw_border(self, xpos, ypos, color):
        x = self._pixel(xpos)
        y = self._pixel(ypos)

        draw = ImageDraw.Draw(self.canvas)
        draw.rectangle((x-2, y-2, x+self.image_size + 2, y+self.image_size + 2), color)
        del draw

    def paste_np(self, arr, xpos, ypos, **kwargs):
        """Paste an np array."""
        if self.mode == 'L':
            shape = [self.image_size, self.image_size]
        elif self.mode == 'RGB':
            shape = [self.image_size, self.image_size, 3]
        elif self.mode == 'RGBA':
            shape = [self.image_size, self.image_size, 4]
        else:
            raise ValueError('Invalid mode.')

        arr = np.reshape(arr, shape)
        im = Image.fromarray(arr, self.mode)
        im.convert('RGB')
        self.paste(im, xpos, ypos, **kwargs)

    def paste_np_float(self, arr, xpos, ypos, **kwargs):
        """Paste an image whose pixel values are float ranged from [0, 1]."""
        self.paste_np((arr * 255).astype(np.uint8), xpos, ypos, **kwargs)

    def save(self, filename):
        self.canvas.save(filename)

    def show(self):
        self.canvas.show()
