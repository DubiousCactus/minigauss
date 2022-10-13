#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Plot samples from a set prior using mingauss.
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import default_rng

from minigauss import GaussianProcess
from minigauss.priors import ExponentialKernel, PolynomialFunc, Bound


def test_function_1D(x):
    return (x * 6 - 2) ** 2 * np.sin(x * 12 - 4)


NUM_TRAIN_PTS = 20
X_RANGE = (0, 1)

rng = default_rng()
# Prior
# Sample from the known prior to verify that sampling works well
gp = GaussianProcess(
    PolynomialFunc(1, [Bound(0, 10), Bound(0, 10)]),
    ExponentialKernel(sigma_y_bounds=Bound(0, 20)),
)
x = np.sort((rng.uniform(X_RANGE[0], X_RANGE[1], (100, 1))), axis=0)
y_oracle = test_function_1D(x)

y1, mean, mean_var = gp.sample(x)
y2, mean, mean_var = gp.sample(x)
y3, mean, mean_var = gp.sample(x)

# Plotting
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, mean, "-", lw=2, label="GP mean prior")
ax.plot(x, y1, "--", lw=2, label="GP sample prior 1")
ax.plot(x, y2, "--", lw=2, label="GP sample prior 2")
ax.plot(x, y3, "--", lw=2, label="GP sample prior 3")
ax.fill_between(
    x.flatten(),
    mean - 1.96 * np.sqrt(mean_var),
    mean + 1.96 * np.sqrt(mean_var),
    facecolor="lavender",
    label="95% Credibility Interval",
)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("f(x)", fontsize=15)
ax.legend(loc="upper left", prop={"size": 12})
plt.show()

# Training data
x_train = rng.uniform(X_RANGE[0], X_RANGE[1], (NUM_TRAIN_PTS, 1))
y_train = test_function_1D(x_train)
gp.fit(x_train, y_train, n_restarts=10)

y1, mean, mean_var = gp.sample(x)
y2, mean, mean_var = gp.sample(x)
y3, mean, mean_var = gp.sample(x)

# Plotting
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y_oracle, "r--", lw=2, label="Oracle")
ax.plot(
    x_train, y_train, "ro", markerfacecolor="r", markersize=10, label="Training Data"
)
ax.plot(x, mean, "-", lw=2, label="GP mean prior")
ax.plot(x, y1, "--", lw=2, label="GP sample prior 1")
ax.plot(x, y2, "--", lw=2, label="GP sample prior 2")
ax.plot(x, y3, "--", lw=2, label="GP sample prior 3")
ax.fill_between(
    x.flatten(),
    mean - 1.96 * np.sqrt(mean_var),
    mean + 1.96 * np.sqrt(mean_var),
    facecolor="lavender",
    label="95% Credibility Interval",
)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("f(x)", fontsize=15)
ax.legend(loc="upper left", prop={"size": 12})
plt.show()
