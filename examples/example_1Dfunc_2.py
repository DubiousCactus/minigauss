#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Fit a 1D function from noisy observations with minigauss.
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import default_rng

from minigauss import GaussianProcess
from minigauss.priors import ExponentialKernel, ConstantFunc, Bound


def test_function_1D(x):
    return (x * 6 - 2) ** 2 * np.sin(x * 12 - 4)


NUM_TRAIN_PTS = 40
NOISE_STD = 0.9
X_RANGE = (0, 1)


rng = default_rng()
# Oracle
x_oracle = np.sort((rng.uniform(X_RANGE[0], X_RANGE[1], (100, 1))), axis=0)
y_oracle = test_function_1D(x_oracle)

# Posterior
x_train = rng.uniform(X_RANGE[0], X_RANGE[1], (NUM_TRAIN_PTS, 1))
y_train = test_function_1D(x_train)
y_train += NOISE_STD * rng.standard_normal((NUM_TRAIN_PTS, 1))

# Maybe in the future we could do: GaussianProcess(x, y, [PolynomialPrior(deg=2), PolynomialPrior(deg=3), SawToothPrior()])
# and it would optimise for the best prior model as well as hyperparameters.
gp = GaussianProcess(
    ConstantFunc(Bound(0, 10)), ExponentialKernel(sigma_y_bounds=Bound(0, 20))
)
gp.fit(x_train, y_train, n_restarts=10, lr=2e-5)

# GP model predictions
f_sample1, mean, mean_var = gp.predict(x_oracle)
f_sample2, mean, mean_var = gp.predict(x_oracle)
f_sample3, mean, mean_var = gp.predict(x_oracle)

# Plotting
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_oracle, y_oracle, "r--", linewidth=2, label="Oracle")
ax.plot(
    x_train, y_train, "r+", markerfacecolor="r", markersize=10, label="Training Data"
)
ax.plot(x_oracle, mean, "b-", lw=2, label="GP mean prediction")
ax.plot(x_oracle, f_sample1, "g--", lw=2, label="GP sample prediction 1")
ax.plot(x_oracle, f_sample2, "g--", lw=2, label="GP sample prediction 2")
ax.plot(x_oracle, f_sample3, "g--", lw=2, label="GP sample prediction 3")
ax.fill_between(
    x_oracle.flatten(),
    mean - 1.96 * np.sqrt(mean_var),
    mean + 1.96 * np.sqrt(mean_var),
    facecolor="lavender",
    label="95% Credibility Interval",
)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("f(x)", fontsize=15)
ax.set_ylim([-10, 20])
ax.legend(loc="upper left", prop={"size": 12})
plt.show()
