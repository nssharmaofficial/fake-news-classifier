# Fake news detection using logistic regression

This project demonstrates the implementation of a machine learning model to classify news articles as either **True** or **Fake** using logistic regression with PyTorch. The project preprocesses a dataset of news articles, applies text vectorization, and trains a model for binary classification.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)

## Introduction

The goal of this project is to detect fake news by analyzing text data. The model is trained to classify whether a news article is real or fake using logistic regression. The dataset used contains both fake and real news articles, with corresponding labels (1 for real, 0 for fake).

## Installation

### Dependencies
To run the project, you'll need to have the following libraries installed:

```bash
- pandas
- numpy
- torch
- tqdm
- scikit-learn
- nltk
- matplotlib
```

You can install the required packages using pip:

```bash
pip install pandas numpy torch tqdm scikit-learn nltk matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

## Data Preprocessing

1. **Loading the Dataset**: The dataset is divided into two parts, `DataSet_Misinfo_TRUE.csv` and `DataSet_Misinfo_FAKE.csv`, which are concatenated after assigning appropriate labels (1 for true news, 0 for fake news).

2. **Text Preprocessing**: The following preprocessing steps are applied to clean and prepare the text data:
   - Convert to lowercase
   - Remove HTML tags, URLs, hashtags, and mentions
   - Remove punctuation and stop words
   - Remove numbers

3. **Vectorization**: The text data is converted into a Bag-of-Words (BoW) representation using `CountVectorizer` with a maximum of 15,000 features.

## Model Architecture

The model is a simple logistic regression classifier built using PyTorch. It consists of three fully connected layers with ReLU activation functions:

- Input layer: 15,000 features
- Hidden layers: 100 units, 10 units
- Output layer: 2 units (binary classification)

## Training

The model is trained on a dataset split into training and testing sets. The training loop runs for 40 epochs, during which:
- The loss (Cross Entropy) is minimized using Adam optimizer
- The model's performance is evaluated on the training set, and the accuracy is printed at each epoch.

## Evaluation

The model is evaluated on the test set, and the final accuracy is printed. The accuracy achieved in this project was around **95%**.

A loss graph is also plotted to visualize the decrease in loss over the epochs.

## Usage

After training, the model is saved along with the `CountVectorizer`. You can use the `predict` function to classify new articles as either real or fake news.

Example usage:

```python
predict("Your news article text here")
```

The `predict` function will output `1` for real news and `0` for fake news.
