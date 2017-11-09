import logging
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
logging.basicConfig(level=logging.INFO)


gtsrb = GtsrbProvider()
for i in range(10):
    data = gtsrb.next_batch()
    print(data)
