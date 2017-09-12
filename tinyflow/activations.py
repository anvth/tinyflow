import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    A = np.max(0, Z)
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ
