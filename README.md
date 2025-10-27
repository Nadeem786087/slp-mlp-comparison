# Neural Networks from Scratch: Single & Multi-Layer Perceptrons

A comprehensive implementation of both Single-Layer Perceptron (SLP) and Multi-Layer Perceptron (MLP) from scratch using NumPy, featuring comparative analysis on classification tasks.

## Overview

This project implements fundamental neural network architectures with complete educational documentation. It demonstrates the progression from simple linear classifiers to complex non-linear models capable of handling intricate decision boundaries.

## 🏗️ Architecture Implementations

### 1. Single-Layer Perceptron (SLP)
- **Activation Functions**: Step, Sigmoid
- **Learning Rule**: Perceptron update rule
- **Applications**: Linear classification tasks

### 2. Multi-Layer Perceptron (MLP)
- **Architecture**: Input layer, hidden layers, output layer
- **Activation Functions**: Step, Sigmoid, Tanh, ReLU
- **Learning Algorithm**: Backpropagation with gradient descent
- **Applications**: Non-linear classification tasks

## 📊 Dataset Generation

- **Linear Dataset**: 100 samples with linear decision boundary
- **Non-linear Dataset**: 200 samples with circular decision boundary
- **Customizable**: Adjustable dataset size and complexity

## 🎯 Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| Step | `f(z) = 1 if z >= 0 else 0` | Binary classification |
| Sigmoid | `f(z) = 1 / (1 + exp(-z))` | Probabilistic output |
| Tanh | `f(z) = tanh(z)` | Zero-centered output |
| ReLU | `f(z) = max(0, z)` | Hidden layers |

## 🔬 Experimental Framework

### Single-Layer Perceptron Experiments:
| Config | Learning Rate | Epochs | Activation | Final Accuracy |
|--------|---------------|--------|------------|----------------|
| 1 | 0.01 | 50 | Step | 99.0% |
| 2 | 0.1 | 100 | Sigmoid | 100.0% |
| 3 | 0.5 | 200 | Step | 100.0% |

### Multi-Layer Perceptron Features:
- Configurable hidden layers and neurons
- Multiple activation function support
- Backpropagation with chain rule
- Gradient computation for weight updates

## 🛠️ Technical Implementation

### Core Components:
1. **Network Initialization**: Random weight initialization
2. **Forward Propagation**: Layer-wise computation
3. **Backward Propagation**: Gradient computation using chain rule
4. **Weight Updates**: Gradient descent optimization
5. **Training Loop**: Epoch-based learning with accuracy tracking

### Key Algorithms:
```python
# Backpropagation implementation
def backward_propagation(self, X, y, y_pred):
    # Output layer error
    error = y_pred - y
    # Hidden layer error (using chain rule)
    hidden_error = error.dot(self.w2.T) * self.activation_derivative(self.z1)
    # Weight updates
    self.w2 -= self.lr * self.a1.T.dot(error)
    self.w1 -= self.lr * X.T.dot(hidden_error)
