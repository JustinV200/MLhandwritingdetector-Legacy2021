# 🧠 Legacy Neural Network – MNIST Digit Recognizer (July 2021)

**Early pre-coursework machine learning project**, built entirely from scratch in Python.

Created in **July 2021** as a rising high school Junior, my first introduction to machine learning — prior to any formal instruction or exposure to modern ML frameworks like TensorFlow or PyTorch.

This basic feedforward neural network uses only `numpy` and `scipy` to train on a subset of the **MNIST handwritten digit dataset**, using backpropagation and sigmoid activation.

---

## 📌 Overview

- **Architecture**:
  - Input Layer: 784 nodes (28×28 grayscale pixels)
  - Hidden Layer: 200 nodes
  - Output Layer: 10 nodes (digits 0–9)
- **Activation Function**: Sigmoid 
- **Training**: Manual backpropagation
- **Data**: CSV versions of MNIST
- **Dependencies**: `numpy`, `scipy`

---

## 📂 Files

- `Network` class: Defines the neural network and learning logic
- `learn()` method: Performs backpropagation training
- `predict()` method: Queries the network for digit recognition
- Training loop and test loop with performance output

---

## 📈 Performance

- Trained on `mnist_train_100.csv`
- Evaluated on `mnist_test_10.csv`
- Achieves basic digit classification with decent performance for a self-coded model

---

## 🛠️ How to Run

1. Install dependencies:
   ```bash
   pip install numpy
   pip install scipy
