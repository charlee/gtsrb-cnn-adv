Adversarial Examples
========================

## Data List

- `mnist_adv_fgsm`: FGSM examples on MNIST dataset CNN.
- `mnist_adv_jsma`: JSMA examples on MNIST dataset CNN.
- `gtsrb_adv_fgsm`: FGSM examples on GTSRB dataset CNN.
- `gtsrb_adv_jsma`: JSMA examples on GTSRB dataset CNN.

## Files

Adversarial examples are stored as PNG files. File format is:
- MNIST: 10 rows x 10 columns, 28px per row/column, 8px gap between rows/columns
- GTSRB: 21 rows x 21 columns, 32px per row/column, 8px gap between rows/columns.
  (Only picked up half classes, i.e. class 0, 2, 4, ..., 42)

Each row represents a source class while echo column represents a target column.

We randomly picked one training example per class from the training set and tried to 
generate adversarial examples for other classes.

- Images with RED borders are the original training example.
- Images with BLUE borders successfully confused the CNN (CNN could not make a correct predict)
  but failed to be identified as given target.
- Images with GREEN borders are successfully predicted as target class.
- Images without bordes failed to find adversarial examples.

## Data

For MNIST dataset, adversarial examples are generated under the epsilons of
`[0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
PNG files are suffixed with `int(eps * 100)` so `adv_examples-02.png` means `eps = 0.02`.

There's another file called `success_rate.npy` which records the success rate for each
source - target pair, i.e. how much `eps` is required for a successful targetd attack.
Its an `ndarray` dump and can be loaded with `np.load('success_rate.npy')`.

For GTSRB dataset, only one set of parameters `theta=1, gamma=0.1` is used due to limited time.

## Observation

[TODO: need data of eps -> successful rate]

For FGSM, `eps >= 0.3` can be easily detected by human. Untargetd attack has a decent success
rate when `eps == 0.3` while targeted attack has the best success rate around `eps == 0.4`.
However `eps` greater than `0.4` will make targeted attack less successfull.

Therefore practically, `eps` should be chosen between `0.1~0.3`.

This is true for both GTSRB and MNIST dataset. However, for untargeted attack, a decent 
success rate can be achieved when `eps == 0.1` while the same `eps` does not work for MNIST.
This is likely because of MNIST has sharp edge ahd high contrast which makes small `eps`
less successful.

For JSMA, attemped setting `theta=1, gamma=0.1` worked better on MNIST than on GTSRB but more
detectable on GTSRB. This is considered as the nature of different contrast.
