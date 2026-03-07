Wi-DMAR: Cross-Domain Human Activity Recognition via an Enhanced Conditional Diffusion Model
This is the PyTorch source code for Wi-DMAR, a WiFi CSI-based cross-domain human activity recognition framework. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands.

Dataset
The two public datasets used in the paper are shown below.
Widar3.0 Dataset
Widar3.0 is one of the most widely used datasets in wireless-sensing-based HAR applications. It covers six basic human–computer interaction gestures (push and pull, sweep, clapping, slide, circle drawing, and zigzag drawing) with 12,750 samples, as well as ten semantic gestures corresponding to digits 0–9 with 5,000 samples. The dataset can be downloaded from the official source:

Widar3.0 Dataset

SignFi Dataset
The SignFi dataset covers 276 frequently used American Sign Language gestures with 8,280 samples collected across both laboratory and home environments:

SignFi Dataset


Requirements

Python 3.7
PyTorch
MATLAB (for data preprocessing)
NumPy
scikit-learn

The codes are tested under Windows 10.

Folder Descriptions

01DataProcessing: This is used to extract CSI data from raw WiFi signals, apply Hampel + Moving Average filtering, Butterworth low-pass filtering, PCA, and mutual information computation to select the most informative subcarriers, and convert the results into a format suitable for model training (Data Preprocessing.m).
02DataAugmentation: This is used to generate augmented pseudo samples that resemble the target-domain distribution using a conditional diffusion model. Data Augmentation-v1.py implements the core diffusion-based augmentation pipeline. Data Augmentation-v2.py builds upon v1 and additionally incorporates domain consistency loss and domain-guided diffusion loss for improved cross-domain alignment.
03ActivityRecognition: This is used to conduct cross-domain activity recognition using a hybrid CNN–BiLSTM network trained with triplet loss, and performs inference via a support-set-based nearest-neighbor decision mechanism (Activity Recognition.py).
