"""
Canvas class used to paste images.
"""

import numpy as np
from PIL import Image

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

        if mode == 'L':
            color = 255
        elif mode == 'RGB':
            color = (255, 255, 255)
        elif mode == 'RGBA':
            color = (255, 255, 255, 1)
        else:
            raise ValueError('Invalid mode {}.'.format(mode))

        self.canvas = Image.new(mode, (self.canvas_w, self.canvas_h), color=color)

    def _pixel(self, p):
        return (self.image_size + self.gap_size) * p + self.gap_size

    def paste(self, im, xpos, ypos):
        """Paste im to (xpos, ypos) position."""
        x = self._pixel(xpos)
        y = self._pixel(ypos)
        self.canvas.paste(im, (x, y))

    def paste_np(self, arr, xpos, ypos):
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
        im = Image.fromarray(arr)
        self.paste(im, xpos, ypos)

    def save(self, filename):
        self.canvas.save(filename)

    def close(self):
        self.canvas.close()
