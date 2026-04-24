# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a neural network that learns to prune its own weights during training using learnable sigmoid gates and L1 regularization.

## Features

* Custom PrunableLinear layer
* Automatic weight pruning during training
* Sparsity vs Accuracy trade-off analysis

## How to Run

pip install torch torchvision matplotlib numpy
python main.py

## Results

The model achieves ~52% accuracy with ~30% sparsity, demonstrating effective pruning.

## Outputs

* Gate distribution plot
* Sparsity vs accuracy graph
