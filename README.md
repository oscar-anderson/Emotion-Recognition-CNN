# Emotion-Recognition Convolutional Neural Network

This repository contains a Python script that implements a Convolutional Neural Network capable of identifying emotions from input face image data. The network was developed using Keras and trained on the FER-2013 dataset of expressive face images. The purpose of this script was to allow me to practice my skills in producing clean Python code, as well as develop my understanding in the field of deep learning. Completion of this project was achieved using a number of online tutorials and documentation.

## Overview

Convolutional Neural Networks (CNNs) are a type of deep learning model that use convolutional layers to apply filters (aka. kernels) to input data, to extract visual features such as those comprising edges, textures, and patterns. The network then learns by iteratively adjusting the parameters of these filters during training, optimising its ability to recognise and discriminate between different features (i.e. 'learning'). In the case of this project, our CNN learns the features of different emotions through training on input expressive face images (taken from the FER-2013 dataset), then applies this learning to classify emotions from new input face data.

## Contents

### README.md
-This file.

### Code/
-This folder contains the Python script [`emotion-recognition-CNN.py`](https://github.com/oscar-anderson/Emotion-Recognition-CNN/blob/main/code/emotion_recognition_cnn.py), which implements the Emotion-Recognition Convolutional Neural Network. The script consists of the following functions:
- `load_data`: This function loads and preprocesses the training and testing data from the input respective folder paths. In the case of the training data, this is also augmented - so as to introduce variability in the training data, to ensure the network's learning is generalisable to a wider range of input testing data.
- `create_CNN`: This function defines the architecture of the Convolutional Neural Network, comprising two convolutional layers with max-pooling, followed by a flattening layer and two dense layers. The final output layer uses the softmax activation function to output the probabilities for each emotion category.
- `train_model`: This function trains the CNN on the FER-2013 training data. Specifically, it takes batches of the training dataset and feeds this into the model. The model then updates its weights and biases through learning with this training data over a specified number of epochs.
- `evaluate_model`: This function evaluates the trained model on the testing data, computing the metrics of loss and accuracy.
- `plot_performance`: This function then generates plots visualising model's accuracy and loss throughout training and validation, over the epochs.

## Usage
1. Ensure you have Python and the relevant dependencies (keras, matplotlib) installed:

   ```
   pip install keras matplotlib
   ```
   
2. Ensure you have the FER-2013 [dataset](https://www.kaggle.com/datasets/msambare/fer2013/data) installed.


3. Clone the repository:

  ```
  git clone https://github.com/oscar-anderson/emotion-recognition-cnn.git
  ```

4. Run the script:

  ```
  cd emotion-recognition-cnn
  python emotion-recognition-cnn.py
  ```

## Acknowledgements
- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)
- [FER-2013 dataset (shared to Kaggle.com by user Manas Sambare)](https://www.kaggle.com/datasets/msambare/fer2013/data)
