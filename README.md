# KNN Breast Cancer Classification

This repository contains a simple implementation of the K-Nearest Neighbors (KNN) algorithm to classify breast cancer as malignant or benign based on a dataset. The project includes preprocessing, training, testing, and prediction steps.

## Features
- Implements KNN from scratch without libraries.
- Handles a CSV dataset of cancer diagnostics.
- Includes Euclidean distance calculation and accuracy evaluation.
- Allows user input for custom predictions.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`

## Dataset
The dataset used is "Cancer_Data.csv," which includes diagnostic results and features for breast cancer cases.

## How it Works
1. Preprocesses the dataset by cleaning and converting labels (malignant: `1`, benign: `0`).
2. Splits the data into training and testing sets.
3. Uses the KNN algorithm to classify test samples.
4. Computes accuracy and allows custom predictions.

## Output
- Classification accuracy on the test data.
- Prediction for user-provided input sample.


