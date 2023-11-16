# Deep Learning Image Classification

## Overview
This GitHub repository contains a Jupyter notebook implementing a deep learning model for image classification using **PyTorch**. The model is based on a modified ResNet18 architecture, trained and validated on a dataset for binary classification. The notebook tracks key performance metrics over epochs, providing insights into the model's training process.

## Dependencies
- **PyTorch**: Deep learning library
- **torch.nn, torch.optim**: PyTorch modules for neural networks and optimization
- **torch.utils.data**: PyTorch module for data loading and preprocessing
- **torchvision.transforms**: Image data transformations
- **torchvision.models**: Pre-trained models from torchvision
- **PIL (Pillow)**: Image manipulation library
- **NumPy**: Numerical computing library
- **pandas**: Data manipulation library
- **matplotlib**: Plotting library for visualization
- **random**: Python module for random number generation

## Model Architecture
The notebook defines a custom neural network class, `Resnet50`, based on the ResNet18 architecture. The classifier is adapted for binary classification with a specified number of classes.

## Training
The model is trained using the **Adam optimizer** and **CrossEntropyLoss**. The training loop runs for a set number of epochs, tracking training and validation loss, as well as accuracy. The training process is detailed for each epoch.

## Visualization
The notebook includes visualizations of training and validation loss over epochs using **matplotlib**. Two subplots display loss and accuracy trends throughout the training process.

**Note**: 
- The notebook may include warnings about deprecated functionalities related to torchvision models, which are safe to ignore.
- The code assumes a binary classification task with two classes.
- Ensure that the required dependencies are installed before running the notebook.
"""
