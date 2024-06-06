# FACE_RECOGNITION
transfer learning using MobileNetV2 and celebA dataset


Sure! Here is a sample README file for your project:

---

# Face Recognition Model Using MobileNetV2

This repository contains the implementation of a face recognition model using transfer learning from MobileNetV2, trained on a subset of the CelebA dataset. The model is designed to run on a GPU if available.


## Overview

The goal of this project is to create a face recognition model using a pre-trained MobileNetV2 architecture. The model is fine-tuned on a subset of the CelebA dataset to recognize faces. The code includes support for GPU acceleration to speed up training and inference.

## Dataset

The model is trained on a limited subset of the CelebA dataset. CelebA is a large-scale face attributes dataset with more than 200,000 celebrity images, each annotated with 40 attribute labels. For this project, a subset of 1000 images is used for training, validation, and testing.

## Model Architecture

The face recognition model uses MobileNetV2 as the backbone. The final fully connected layer is replaced to match the number of classes in the dataset.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Torchvision 0.8+
- Matplotlib
- CUDA (for GPU support)
- CelebA dataset (automatically downloaded by the code)

Install the required packages using pip:

```bash
pip install torch torchvision matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-recognition-mobilenetv2.git
   cd face-recognition-mobilenetv2
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

The script will automatically download the CelebA dataset, create the necessary data loaders, and train the model.

## Training and Evaluation

The training script performs the following steps:

1. Loads the CelebA dataset and creates subsets for training, validation, and testing.
2. Defines data transformations including resizing, normalization, and data augmentation for training data.
3. Initializes the MobileNetV2 model and modifies the final layer.
4. Moves the model and data to the GPU if available.
5. Trains the model and logs the training and validation losses.
6. Evaluates the model on the test set and prints the accuracy.
7. Saves the trained model to a file.
8. Displays a few test images along with their ground truth and predicted labels.

## Results

The script outputs the training and validation losses for each epoch and the final test accuracy. It also plots the training and validation losses over the epochs and displays some test images with their predictions.


