# Wi-DMAR: Cross-Domain Human Activity Recognition via an Enhanced Conditional Diffusion Model

This is the PyTorch source code for Wi-DMAR, a WiFi CSI-based cross-domain human activity recognition framework. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

## Dataset

The two public datasets used in the paper are shown below.

### Widar3.0 Dataset

The Widar3.0 dataset comes from the link below: https://tns.thss.tsinghua.edu.cn/widar3.0/

It is also available via:
- IEEE DataPort: https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
- Baidu Disk (password: 4m47): https://pan.baidu.com
- FTP (FileZilla client recommended): `166.111.80.127:40121` (username: widarftp, password: widar2019)

### SignFi Dataset

The SignFi dataset comes from the link below: https://github.com/yongsen/SignFi

## Requirement

Python 3.7

PyTorch

MATLAB

The codes are tested under Windows 10.

## Folder descriptions:

*Data Preprocessing.m*: This is used to extract raw CSI data, apply Hampel filter, moving average filter, and Butterworth low-pass filter for denoising, then perform PCA combined with mutual information computation to select the top 30 most discriminative subcarriers.

*Data Augmentation-v1.py*: This is used to implement the core conditional diffusion-based data augmentation pipeline, including the domain feature extraction module and the generation module.

*Data Augmentation-v2.py*: This is built upon *Data Augmentation-v1.py* and further incorporates domain consistency loss and domain-guided diffusion loss to generate pseudo samples that closely match the target-domain distribution.

*Activity Recognition.py*: This is used to conduct cross-domain activity recognition using a hybrid CNN-BiLSTM network trained with triplet loss, and performs nearest-neighbor inference over a 5-shot support set.
