import numpy as np
from cnn_gtsrb.dataset.canvas import Canvas
from PIL import Image


class AdversarialExampleReader():
    def dump_adv(self, filename, count=10):
        data = np.load(filename)
        image_size = int(data[0][0])
        channels = int(data[0][2])

        if channels == 3:
            mode = 'RGB'
        else:
            mode = 'L'

        canvas = Canvas(image_size, 3, count, mode=mode)

        print('Num of examples: {}'.format(data.shape[0]))
        print('Image size: {}'.format(image_size))
        print('Classes: {}'.format(data[0][1]))
        print('Channels: {}'.format(data[0][2]))

        for i, row in enumerate(data[:count]):
            image_data_size = image_size * image_size * channels
            image = row[10:10+image_data_size]
            adv = row[10+image_data_size:10+image_data_size*2]
            pertubation = adv - image

            legit_predict = row[7]
            target_predict = row[8]
            adv_predict = row[9]

            if adv_predict == legit_predict:
                border = 0
            elif adv_predict == target_predict:
                border = 2
            else:
                border = 3

            print("#{}: param={}, label/pred/target/adv_pred = {}".format(i, row[2:4], row[6:10]))

            canvas.paste_np(image.astype(np.uint8), 0, i)
            canvas.paste_np(np.clip(pertubation, 0, 255).astype(np.uint8), 1, i)
            canvas.paste_np(adv.astype(np.uint8), 2, i, border=border)

        canvas.show()
