README.md
=============

## Preparation

First setup python environment.

```python
> virtualenv .env
> .env/Scripts/activate
(.env) > pin install -r requirements-gpu.txt
```

## Train a CNN for GTSRB

To generate adversarial examples for GTSRB, first run the script below to train a CNN model for GTSRB dataset.

```
python gtsrb_cnn_run.py
```

This script will download the dataset from GTSRB website, extract it and generate training set for the CNN.

Then it will create a basic CNN and train it with 20000 steps, each step uses 100 examples.

You should be able to achieve about 93% test accuracy.

The trained model will be saved under `tmp/gtsrb_model-32x32` and can be viewed with `tensorboard`.

## Adversarial Example Crafting

Run the following command:

```python
python gtsrb_jsma_batch_crafting.py
```

This will run JSMA attack on 5000 test examples and generate adversarial examples for each.

Generated examples are saved under `tmp/batch_adv`.

