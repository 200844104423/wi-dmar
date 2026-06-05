# Wi-DMAR: Cross-Domain Human Activity Recognition via an Enhanced Conditional Diffusion Model

This is the PyTorch source code for Wi-DMAR, a WiFi CSI-based cross-domain human activity recognition framework. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

## Dataset

The two public datasets used in the paper are shown below.

### Widar3.0 Dataset

The Widar3.0 dataset comes from the link below: https://tns.thss.tsinghua.edu.cn/widar3.0/

### SignFi Dataset

The SignFi dataset comes from the link below: https://github.com/yongsen/SignFi

## Requirement

Python 3.7

PyTorch

MATLAB

## Folder descriptions:

*Data Preprocessing.m*: This is used to preprocess raw CSI data and select discriminative subcarriers.

*Data Augmentation-v1.py*: This is used to implement the core modules for conditional diffusion-based data augmentation.

*Activity Recognition.py*: This is used to conduct activity recognition.
