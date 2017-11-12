import numpy as np
from cnn_gtsrb.dataset.canvas import Canvas
from PIL import Image


class AdversarialExampleReader():
    def dump_adv(self, filename, count=10):
        data = np.load(filename)
        image_size = int(data[0][0])
        canvas = Canvas(image_size, 3, count)

        print('Num of examples: {}'.format(data.shape[0]))
        print('Image size: {}'.format(image_size))
        print('Classes: {}'.format(data[0][1]))
        print('params: {}'.format(data[:count,2:6]))

        for i, row in enumerate(data[:count]):
            image_data_size = image_size * image_size
            image = row[10:10+image_data_size]
            adv = row[10+image_data_size:10+image_data_size*2]
            pertubation = adv - image

            canvas.paste_np(image.astype(np.uint8), 0, i)
            canvas.paste_np(np.clip(pertubation, 0, 255).astype(np.uint8), 1, i)
            canvas.paste_np(adv.astype(np.uint8), 2, i)

        canvas.show()
