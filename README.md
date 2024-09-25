# Kolmogorov–Arnold Neural Network (KANN)


![kann](https://github.com/user-attachments/assets/902b7fa4-58c7-4f35-8f1a-d96a80c587d0)


This repository contains an implementation of a Kolmogorov–Arnold Neural Network (KANN) to approximate a target function using TensorFlow and NumPy. 

The network is designed to learn and approximate this function over a grid of input values. The results are visualized with a 3D surface plot comparing the original function and the neural network's approximation.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)


## Introduction

Kolmogorov–Arnold neural networks are used to approximate multivariable functions by decomposing them into sums of simpler functions. In this implementation, the hidden layer applies summations of simple functions \(\psi\), followed by an output layer applying a \(\phi\) function.

This specific example demonstrates the network's ability to approximate a simple trigonometric function, but the approach can be generalized to more complex cases.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib



## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/kann-approximation.git
cd kann-approximation
pip install -r requirements.txt

