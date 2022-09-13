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
from minigauss.priors import ExponentialKernel, PolynomialFunc


X_RANGE = (-1, 1)

rng = default_rng()
# Prior
# Sample from the known prior to verify that sampling works well
gp = GaussianProcess(PolynomialFunc(2), ExponentialKernel())
x = np.sort((rng.uniform(X_RANGE[0], X_RANGE[1], (200, 1))), axis=0)
y1, mean, mean_var = gp.sample(x)
y2, mean, mean_var = gp.sample(x)
y3, mean, mean_var = gp.sample(x)


# Plotting
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, mean, "b-", lw=2, label="GP mean prior")
ax.plot(x, y1, "g--", lw=2, label="GP sample prior 1")
ax.plot(x, y2, "g--", lw=2, label="GP sample prior 2")
ax.plot(x, y3, "g--", lw=2, label="GP sample prior 3")
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
