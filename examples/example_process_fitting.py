#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Fit a 1D random process from noisy observations of several realizations with minigauss.
Most tutorials focus on fitting functions with GPs, but I never see them attempting to fit random
processes. Isn't that the actual goal of GPs though?
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import default_rng

from minigauss import GaussianProcess
from minigauss.priors import Bound, ExponentialKernel
from minigauss.priors.mean import ConstantFunc

NUM_TRAIN_REALIZATIONS = 6
NUM_TRAIN_PTS_PER_REALIZATION = 60
MAX_NUM_OBS_PTS = 40
X_RANGE = (0, 5)
NUM_TARGET_PTS = 400

""" TRAINING """
rng = default_rng()
# Oracle process (a preset GP)
orcale_gp = GaussianProcess(
    ConstantFunc(value=0), ExponentialKernel(sigma_y=2.3, l=0.4, sigma_n=0.8)
)  # Add a little bit of noise

# Training data: in practice, you may have arrays of measurements from independent realizations of
# a process.
y_train = np.zeros((NUM_TRAIN_REALIZATIONS, NUM_TRAIN_PTS_PER_REALIZATION, 1))
x_train = np.zeros((NUM_TRAIN_REALIZATIONS, NUM_TRAIN_PTS_PER_REALIZATION, 1))
for i in range(NUM_TRAIN_REALIZATIONS):
    x_train[i, :] = np.sort(
        (rng.uniform(X_RANGE[0], X_RANGE[1], (NUM_TRAIN_PTS_PER_REALIZATION, 1))),
        axis=0,
    )
    y, _, _ = orcale_gp.sample(x_train[i])
    y_train[i, :] = np.expand_dims(y, axis=1)  # TODO: Return same dim as input!

# Defining our GP that we want to fit to the data (to hopefully learn the oracle hyperparameters)
# Should fit: ConstantFunc(value=7), ExponentialKernel(sigma_y=2.3, l=0.4, sigma_n=0.1)
gp = GaussianProcess(ConstantFunc(), ExponentialKernel(), use_scipy=False)
# Training the gp: to keep things simple in the library, let's merge all data points into one
# training set.
# TODO: Implement mini-batch training
gp.fit(
    np.vstack(x_train),
    np.vstack(y_train),
    n_restarts=10,
    max_fast_iterations=50,
    lr=1e-4,
    decay_rate=0.99
)


""" OBSERVATIONS """
# Now let's assume we have measurements from a new realization of the process. We can condition our
# fitted GP onto those to predict other target values! Let's look at 4 distinct realizations, and
# you'll see the TRUE power of the GP!
# creates two subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
for i, ax in enumerate(np.hstack(axes)):
    x_rlz = np.sort(rng.uniform(X_RANGE[0], X_RANGE[1], (NUM_TARGET_PTS, 1)), axis=0)
    y_rlz, _, _ = orcale_gp.sample(
        x_rlz
    )  # One realization of the true process we want to model. This wasn't seen during training.
    # Pick some observations from this process realization:
    n_obs = np.random.randint(3, MAX_NUM_OBS_PTS)
    idx_obs = np.sort(np.random.permutation(NUM_TARGET_PTS)[:n_obs], axis=0)
    x_obs = x_rlz[idx_obs]
    y_obs = np.expand_dims(y_rlz[idx_obs], axis=1)

    x_tgts = np.sort(rng.uniform(X_RANGE[0], X_RANGE[1], (NUM_TARGET_PTS, 1)), axis=0)

    # GP model predicting
    gp.observe(x_obs, y_obs)  # Recomputes K, K_inv, mu and set gp._x, gp._y
    f_sample1, mean, mean_var = gp.predict(x_tgts)
    f_sample2, mean, mean_var = gp.predict(x_tgts)
    f_sample3, mean, mean_var = gp.predict(x_tgts)

    # Plotting
    ax.plot(x_rlz, y_rlz, "r--", linewidth=2, label=f"Oracle process realization {i}")
    ax.plot(
        x_obs,
        y_obs,
        "r+",
        markerfacecolor="r",
        markersize=10,
        label=f"Observations ({n_obs})",
    )
    ax.plot(x_tgts, mean, "b-", lw=2, label="GP mean prediction")
    ax.plot(x_tgts, f_sample1, "g--", lw=2, label="GP sample prediction 1")
    ax.plot(x_tgts, f_sample2, "g--", lw=2, label="GP sample prediction 2")
    ax.plot(x_tgts, f_sample3, "g--", lw=2, label="GP sample prediction 3")
    ax.fill_between(
        x_tgts.flatten(),
        mean - 1.96 * np.sqrt(mean_var),
        mean + 1.96 * np.sqrt(mean_var),
        facecolor="lavender",
        label="95% Credibility Interval",
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("f(x)", fontsize=15)
    ax.legend(bbox_to_anchor=(1, 1), prop={"size": 12}, loc="upper left")
plt.subplots_adjust(right=0.85, wspace=0.7)
plt.show()
