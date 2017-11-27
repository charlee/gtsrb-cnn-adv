import glob
import numpy as np


for theta in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    filename = 'fgsm_mnist_bg-{:0.2f}-*.npy'.format(theta)


    results = []
    for f in glob.glob('tmp/adv_mnist_bg-28x28/' + filename):
        results.append(np.load(f))
    # results.append(np.load('tmp/batch_adv/jsma_gtsrb-0.2-0.npy'))
    
    data = np.concatenate(results, axis=0)

    actual_theta = np.unique(data[:,3])[0]

    # filter out target == original label
    data = data[data[:,6] != data[:,8]]
        
    total = data.shape[0]

    # misclassified count
    misclassified = np.sum(data[:,6] != data[:,9])
    # targeted
    targeted = np.sum(data[:,8] == data[:,9])

    print('{}, {}, {}, {}'.format(actual_theta, total, misclassified / total, targeted / total))