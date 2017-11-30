import glob
import numpy as np


results = []
for f in glob.glob('tmp/adv_mnist-28x28/fast-jsma*.npy'):
    results.append(np.load(f))

data = np.concatenate(results, axis=0)

# filter out target == original label
data = data[data[:,6] != data[:,8]]
    
total = data.shape[0]

for gamma in [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]:
    misclassified = np.sum(data[:,4] < gamma)
    targeted = np.sum(data[:,5] < gamma)

    print('{}, {}, {}, {}'.format(gamma, total, misclassified / total, targeted / total))