# SPRT-TANDEM-Project
___Cite our paper and this page if you use the whole or a part of our code.___
```
Bibtex:

@article{XXX
  XXX
}
@misc{XXX
  XXX
}
```

## Requirements
- TensorFlow 2.0.0

## Files and Directories
- `train_fe_nmnist.py`
  - training script for feature extractor (ResNet) on Nosaic MNIST
- `train_ti_nmnist.py`
  - training script for temporal integrator (LSTM) on Nosaic MNIST
- `make_nmnist.py`
  - Use this file to make Nosaic MNIST dataset.
- `demo_nmnist.ipynb`
  - Nosaic MNIST is like this. Have a look!
- `configs/config_fe_nmnist.yaml`
  - config fiel for `train_fe_nmnist.py`
- `configs/config_ti_nmnist.yaml`
  - config file for `train_ti_nmnist.py`
- `datasets/data_processing.py`
  - data loader and data processing functions
- `models/backbones_fe.py`
  - ResNet v1 and v2 models
- `models/backbones_ti.py`
  - LSTM model
- `models/losses.py`
  - loss functions and gradient computation. 
  - ___LLLR is in `binary_llr_loss_func`.___
- `models/optimizers.py`
  - optimizer switching function
- `utils/misc.py`
  - miscellaneous minor functions
- `utils/performance_metrics.py`
  - performance metrics such as balanced accuracy 
  - ___the SPRT algorithm is in `binary_truncated_sprt`.___
- `utils/util_ckpt.py`
  - utility functions to save checkpoints (models)
- `utils/util_optuna.py`
  - utility functions for Optuna
- `utils/util_tensorboard.py`
  - TensorBoard logger class

## Anonymous Github Spec
https://anonymous.4open.science/r/4dc2b523-af99-4f70-a10f-d6c544d58c21/

NG words (add affiliations, author names, and ...):

Akinori F. Ebihara

Akinori-F-Ebihara

Akinori

Ebihara

Taiki Miyagawa

TaikiMiyagawa

Taiki

Miyagawa

t-miyagawa

taiki

miyagawa

Kazuyuki Sakurai

Kazuyuki 

Sakurai

Hitoshi Imaoka

Hitoshi

Imaoka

RIKEN

AIP

NEC

