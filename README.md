# Multiclass Neural Network for Handwritten Digit Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![Keras](https://img.shields.io/badge/Keras-Sequential-red?logo=keras) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project implements a **multiclass neural network classifier** to recognize handwritten digits (0–9) using the **MNIST dataset** — one of the most well-known benchmarks in machine learning and computer vision.

The model takes a 28×28 grayscale image of a handwritten digit and predicts which digit (0 through 9) it represents. Handwritten digit recognition is a foundational problem in pattern recognition with real-world applications in postal sorting, bank cheque processing, and digitizing handwritten forms.

The project focuses not only on training a high-performing model with Keras, but also on deeply understanding the underlying mathematics — including a **custom softmax implementation from scratch** verified against TensorFlow's built-in function.

---

## ✨ Key Features

- **Data Preprocessing** — Pixel values normalized to the [0, 1] range and images flattened from 28×28 to 784-dimensional vectors
- **Neural Network Training** — A three-layer fully connected network trained with the Adam optimizer over 40 epochs
- **Custom Softmax Implementation** — Manual NumPy implementation of the softmax function, validated against `tf.nn.softmax`
- **Loss Monitoring** — Training loss tracked across all epochs, demonstrating convergence from ~0.24 down to ~0.007
- **Prediction Visualization** — A 5×5 grid of test samples displaying predicted vs. true labels, color-coded green (correct) and red (incorrect)
- **Model Inspection** — Weight and bias shapes extracted and verified for each layer to confirm the architecture

---

## 🧠 Machine Learning Concepts Implemented

| Concept | Description |
|---|---|
| **Multiclass Classification** | Predicts one of 10 digit classes (0–9) for each input image |
| **Feedforward Neural Network** | Three Dense layers with ReLU activations and a linear output layer |
| **Softmax Activation** | Converts raw logits into class probability distributions; implemented both manually and via TensorFlow |
| **Sparse Categorical Cross-Entropy** | Loss function used for integer-encoded multi-class labels with `from_logits=True` |
| **Adam Optimizer** | Adaptive learning rate optimizer (lr=0.001) used for efficient gradient-based training |
| **Feature Normalization** | Dividing pixel values by 255 to scale inputs to [0, 1], improving gradient flow and training stability |

---

## 💡 What I Learned

Working through this project gave me hands-on experience with the full pipeline of building a neural network classifier:

- **Implementing softmax from scratch** using NumPy and verifying it against TensorFlow's implementation to build intuition for how probability distributions are generated from logits
- **Understanding SparseCategoricalCrossentropy** and when to use it versus `CategoricalCrossentropy` (integer labels vs. one-hot encoded labels)
- **Configuring a Keras Sequential model** — defining layer sizes, choosing activation functions, and setting `from_logits=True` for numerical stability
- **Monitoring training loss** across epochs to identify convergence and diagnose underfitting or overfitting
- **Visualizing model predictions** on a grid of test images to qualitatively assess model performance beyond a single accuracy metric
- **Inspecting model internals** — extracting weight matrices and bias vectors to verify that layer shapes match the intended architecture

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **NumPy** — array operations and custom softmax implementation
- **Pandas** — data handling utilities
- **Matplotlib** — prediction grid visualization
- **TensorFlow 2.x / Keras** — dataset loading, model definition, training, and evaluation
  - `keras.datasets.mnist`
  - `keras.models.Sequential`
  - `keras.layers.Dense`
  - `keras.losses.SparseCategoricalCrossentropy`
- **Jupyter Notebook** — interactive development environment

---

## 📁 Project Structure

```
multiclass-digit-recognition/
│
├── multiclass_digit_recognization.ipynb   # Main notebook with full implementation
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multiclass-digit-recognition.git
cd multiclass-digit-recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib tensorflow jupyter
```

### 3. Launch the Notebook
```bash
jupyter notebook multiclass_digit_recognization.ipynb
```

Run all cells from top to bottom. The MNIST dataset will be downloaded automatically via `tensorflow.keras.datasets.mnist` on first run.

---

## 📊 Example Output / Results

### Training Loss Convergence
When you run the training cell, you will see the loss printed for each of the 40 epochs:

```
Epoch 1/40  — Loss: 0.2378
Epoch 10/40 — Loss: 0.0421
Epoch 20/40 — Loss: 0.0198
Epoch 40/40 — Loss: 0.0066
```

The steadily decreasing loss confirms the model is learning and converging successfully.

### Softmax Verification
The notebook demonstrates a custom NumPy softmax function and compares its output with `tf.nn.softmax`, confirming numerical equivalence between the manual implementation and TensorFlow's built-in function.

### Prediction Visualization
A **5×5 grid** of test images is rendered with:
- The **predicted label** shown above each image
- The **true label** shown below
- **Green titles** for correct predictions
- **Red titles** for incorrect predictions

This provides a clear, visual sanity check of how well the model generalizes to unseen handwritten digits.

---

## 🚀 Future Improvements

- **Convolutional Neural Network (CNN)** — Replace the dense layers with convolutional layers to leverage the spatial structure of images and achieve higher accuracy
- **Scikit-learn Baseline** — Add a logistic regression or SVM baseline using scikit-learn for comparison against the neural network
- **Regularization** — Experiment with dropout layers or L2 weight regularization to reduce overfitting
- **Data Augmentation** — Apply random rotations, shifts, and zooms during training to improve generalization to real-world handwriting styles
- **Confusion Matrix** — Add a confusion matrix visualization to identify which digit pairs the model most commonly confuses
- **Interactive Web Demo** — Deploy the trained model as a web application where users can draw a digit and receive a live prediction

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
