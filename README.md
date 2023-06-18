# An Empirical Study of Adaptive Data Selection on Real-world Images

## Getting Started

```bash
pip install -r requirements.txt
```

Dataset preparation: download the corresponding datasets and set the data folder in configuration YAML files.

* CIFAR-10 can be downloaded via `torchvision.datasets.CIFAR10()`
* Indoor is accessible on [Indoor Scene Recognition](http://web.mit.edu/torralba/www/indoor.html)
* Kvasir Capsule is accessible on [Simula](https://datasets.simula.no/kvasir-capsule/)
* ISIC 2019 is accessible on [ISIC Challenge](https://challenge.isic-archive.com/data/#2019)

## Adaptive Data Selection

The core data valuation and selection codes are provided in `./data_value/data_value.py`.

Modify the configurations in `config/default.yaml` (including dataset path and training parameters), and run the following command:

```bash
python adaptive_selection.py
```

The random seed is kept the same as in the paper and the train-validation-test split should be the same. The splits are recorded in `split_indices` folder as well for loading them manually. Load the dataset via our `Dataset`s, split (if there are no official splits) train-test by `train.npy` (the training and validation indices), and then split train-validation on the train split by `train_no_val.npy` (the training indices excluding the validation indices).

As for GradMatch, we modify the [official implementation](https://github.com/decile-team/cords).

## Coreset Selection and Verification

1. For the proposed method, run an observation experiment first by setting `observe: True` and `fullset: True` in `config/default.yaml`. Run adaptive data selection and obtain the observed GradNorm scores for each epoch.
2. Run `coreset_selection.py` and obtain the selected coreset indices. The indices will be stored in `./data/coreset` together with those of compared methods. Previous run results are provided in the `./data/coreset` indices folder.
3. Run `adaptive_selection.py` with `config/coreset.yaml` for verification.

As for the other baselines, we select the coresets by modifying and running [DeepCore](https://github.com/patrickzh/deepcore).
