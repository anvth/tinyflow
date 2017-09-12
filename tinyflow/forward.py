import numpy as np
from activations import sigmoid, relu


def linear_forward(A, W, b):
    Z = np.dot(A, W) + b

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    else:
        raise KeyError('No activation function found')

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters['W' + str(l)],
            parameters['b' + str(l)],
            activation='relu'
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A,
        parameters['W' + str(L)],
        parameters['b' + str(L)],
        activation='sigmoid'
    )
    caches.append(cache)

    return AL, caches
