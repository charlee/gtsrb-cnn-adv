import sys
import os

from cnn_gtsrb.attacks.adv_dump import AdversarialExampleReader

ir = AdversarialExampleReader()

if len(sys.argv) <= 1:
    print('Usage: {} <filename>'.format(sys.argv[0]))
    exit(1)

filename = sys.argv[1]
ir.dump_adv(filename)