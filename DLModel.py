"""
This file contains the definition of the neural network used for the Data-driven Ecological Climate Classification (DECC).
For the sake of memory efficiency and simplicity, the inference of the network is implemented in NumPy.
The model can be easily translated to other frameworks if you want to do some future work.

Author: Tracy Van
Date: 2024-12-08
"""

import numpy as np
from numba import njit


@njit
def softmax(x):
    # 计算指数
    exp_x = np.exp(x)
    # 手动计算和并扩展维度
    sum_exp = exp_x.sum(axis=1).reshape(-1, 1)
    return exp_x / sum_exp


# Pad the data of December and January to the beginning and the end of the year
# so that each month is equally represented
def circular_padding(x):
    last_col = x[:, :, -1:]
    first_col = x[:, :, 0:1]
    return np.concatenate([last_col, x, first_col], axis=2)


# Pool across the months
def max_pooling(x):
    return np.max(x, axis=2)


# Pool across the months
def avg_pooling(x):
    return np.mean(x, axis=2)


# The model was originally implemented in MATLAB,
# so the data is flattened in the Fortran order
def flatten(x):
    return x.reshape(x.shape[0], -1, order="F")


@njit
def conv2d(x, kernel, bias):
    batch_size, height, width = x.shape
    output_height = height - 2
    output_width = width - 2
    num_filters = kernel.shape[2]

    output = np.zeros((batch_size, output_height, output_width, num_filters))

    for b in range(batch_size):
        for i in range(output_height):
            for j in range(output_width):
                for k in range(num_filters):
                    patch = x[b, i : i + 3, j : j + 3]
                    output[b, i, j, k] = np.sum(patch * kernel[:, :, k]) + bias[k]

    return output


@njit
def batch_norm(x, coeff, bias):
    return coeff * x + bias


@njit
def linear(x, kernel, bias):
    return x @ kernel + bias


@njit
def pca(x, coeff, mu):
    return (x - mu) @ coeff


@njit
def som(x, centroid):
    distances = np.sum((x[:, np.newaxis, :] - centroid) ** 2, axis=2)
    return softmax(-distances)


# Custom activation function, also serves as data preprocessing
# Compare temperature, precipitation, and PET to trained thresholds and scales, and then tanh
# so that the data is normalized between -1 and 1
@njit
def act(x, temp_mu, temp_sigma, precip_mu, precip_sigma):
    temp = x[:, 0, :]
    precip = x[:, 1, :]
    evap = x[:, 2, :]

    results = np.zeros((x.shape[0], 10, 12))
    results[:, 0, :] = (temp - temp_mu[0]) / temp_sigma
    results[:, 1, :] = (precip - precip_mu[0]) / precip_sigma[0]
    results[:, 2, :] = (evap - precip_mu[0]) / precip_sigma[0]
    results[:, 3, :] = (temp - temp_mu[1]) / temp_sigma
    results[:, 4, :] = (precip - precip_mu[1]) / precip_sigma[1]
    results[:, 5, :] = (evap - precip_mu[1]) / precip_sigma[1]
    results[:, 6, :] = (temp - temp_mu[2]) / temp_sigma
    results[:, 7, :] = (precip - precip_mu[2]) / precip_sigma[2]
    results[:, 8, :] = (evap - precip_mu[2]) / precip_sigma[2]
    results[:, 9, :] = (temp - temp_mu[3]) / temp_sigma

    return np.tanh(results)


class MyActivationLayer:
    def __init__(self, temp_mu, temp_sigma, precip_mu, precip_sigma):
        self.temp_mu = temp_mu
        self.temp_sigma = temp_sigma
        self.precip_mu = precip_mu
        self.precip_sigma = precip_sigma

    def __call__(self, x):
        return act(x, self.temp_mu, self.temp_sigma, self.precip_mu, self.precip_sigma)


class ConvLayer:
    def __init__(self, kernel, bias):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, x):
        return conv2d(x, self.kernel, self.bias)


class BatchNormLayer:
    def __init__(self, beta, gamma, moving_mean, moving_variance, epsilon=1e-5):
        self.coeff = gamma / np.sqrt(moving_variance + epsilon)
        self.bias = beta - self.coeff * moving_mean

    def __call__(self, x):
        return batch_norm(x, self.coeff, self.bias)


class FullyConnectedLayer:
    def __init__(self, kernel, bias):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, x):
        return linear(x, self.kernel, self.bias)


class Network:
    def __init__(self, weight_file, pca_components=15):
        self.act = MyActivationLayer(
            weight_file["act_/temp_mu"][:],
            weight_file["act_/temp_sigma"][:],
            weight_file["act_/precip_mu"][:],
            weight_file["act_/precip_sigma"][:],
        )
        self.conv = ConvLayer(
            weight_file["conv_/kernel"][:].reshape(3, 3, 32),
            weight_file["conv_/bias"][:],
        )
        self.bn = BatchNormLayer(
            weight_file["batchnorm_1_/beta"][:],
            weight_file["batchnorm_1_/gamma"][:],
            weight_file["batchnorm_1_/moving_mean"][:],
            weight_file["batchnorm_1_/moving_variance"][:],
        )
        self.fc1 = FullyConnectedLayer(
            weight_file["fc_1_/kernel"][:].reshape(512, 256),
            weight_file["fc_1_/bias"][:],
        )
        self.bn1 = BatchNormLayer(
            weight_file["batchnorm_/beta"][:],
            weight_file["batchnorm_/gamma"][:],
            weight_file["batchnorm_/moving_mean"][:],
            weight_file["batchnorm_/moving_variance"][:],
        )
        self.fc2 = FullyConnectedLayer(
            weight_file["fc_2_/kernel"][:].reshape(256, 14),
            weight_file["fc_2_/bias"][:],
        )
        self.pca_coeff = weight_file["coeff"][:]
        self.pca_mu = weight_file["mu"][:]
        self.pca_components = pca_components

    def __call__(self, x):
        """
        input shape of x: (batch_size, 3, 12)
        the third dimension refers to 12 months
        """
        x = self.act(x)
        x = circular_padding(x)
        x = self.conv(x)
        x = self.bn(x)
        x[x < 0] = 0  # ReLU
        x1 = flatten(max_pooling(x))
        x2 = flatten(avg_pooling(x))
        x = np.concatenate([x2, x1], axis=1, dtype=np.float32)

        # climate features
        pca_features = pca(x, self.pca_coeff, self.pca_mu)[:, : self.pca_components]

        x = self.fc1(x)
        x = self.bn1(x)
        x[x < 0] = 0  # ReLU, output of this layer can be deemed as biome features
        x = self.fc2(x)
        x = softmax(x)
        # We used binary cross-entropy as the loss function

        # Land Cover data (prediction target) was downloaded from MCD12C1 v061 (https://lpdaac.usgs.gov/products/mcd12c1v061/)
        # We used the IGBP land cover scheme, with the urban and water body types removed, and croplands merged with cropland-natural vegetation mosaics
        return pca_features, x
