# MNIST Digit Recognition Using Whale Optimization Algorithm

## Project Overview

This project implements handwritten digit recognition using a neural network trained on the MNIST dataset. The Whale Optimization Algorithm (WOA) is used to optimize the dropout rate with the aim of improving model performance.

## Technologies Used

- Python
- PyTorch
- NumPy
- Torchvision
- Whale Optimization Algorithm (WOA)
- MNIST Dataset

## Model Architecture

- Input: 28 × 28 pixel grayscale image
- Flatten layer
- Fully connected layer: 128 neurons
- ReLU activation
- Dropout layer (dropout rate optimized using WOA)
- Output layer: 10 classes (digits 0–9)

## Features

- MNIST dataset downloading through Torchvision (when the training script is configured to download it)
- WOA-based dropout-rate optimization
- Neural network training
- Model evaluation
- Handwritten digit prediction

## Installation

Clone or download this repository, then install the required packages:

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python train_model.py
```

## Run Prediction

```bash
python predict.py
```

## Project Files

- `train_model.py` — trains the neural network and performs dropout-rate optimization, if implemented in this script.
- `predict.py` — loads the trained model and predicts a digit from an input image.
- `requirements.txt` — lists the Python dependencies.
- `mnist_model.pth` — saved model weights (include this file in the repository if you want users to run prediction without training first).

## Notes

- Run commands from the project directory.
- Ensure the model architecture and saved weights used by `predict.py` match.
- If your training script has a different filename, replace `train_model.py` in the training command with the actual filename.
