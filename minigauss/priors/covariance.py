# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Covariance priors.
"""

import abc
import numpy as np

from typing import Any, Dict, Optional

from . import Prior, Bound


class CovariancePrior(Prior, metaclass=abc.ABCMeta):
    def __init__(
        self, bounded_params: Dict[str, Bound], set_params: Dict[str, Any] = {}
    ) -> None:
        # Every covariance kernel should have a noise std parameter
        bounded_params["sigma_n"] = Bound(1e-3, 1)
        super().__init__(bounded_params, set_params)

    @abc.abstractmethod
    def _covariance_mat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray, y: np.ndarray, observations=False) -> np.ndarray:
        K = self._covariance_mat(x, y)
        if observations:
            K += np.eye(x.shape[0]) * self.sigma_n**2
        return K

    def optimize_one_step(
        self,
        ctx: dict,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        optimizer="gradient_ascent",
    ) -> None:
        assert "mu" in ctx, "mu not set in context!"
        # Only add noise to the diagonal because we assume the observations to have independent
        # additive gaussian noise. The noise is added to the variance of the random variables,
        # not the covariances!!
        K = self(x, x) + self.sigma_n**2 * np.eye(x.shape[0])
        K_inv = np.linalg.inv(K)
        residuals = y - ctx["mu"]
        self._compute_gradients(x, y)
        if optimizer == "gradient_ascent":
            for param in self._params.keys():
                dK_dparam = self._grads[param]
                K_inv_dot_dK_dparam = K_inv @ dK_dparam
                lml_pd = 0.5 * (
                    np.trace(K_inv_dot_dK_dparam)
                    + (residuals.T @ dK_dparam @ K_inv_dot_dK_dparam @ residuals).item()
                )
                # Regularize this hyperparam because gradient ascent will definitely cheat
                if param == "sigma_n":
                    lml_pd *= 1e-5
                self._params[param] += lr * lml_pd
                # To help the optimization process, since those values should always be
                # non-negative
                self._params[param] = abs(self._params[param])
        else:
            # TODO: More methods! https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/
            raise NotImplementedError(f"Optimizer {optimizer} is not implemented.")


class ExponentialKernel(CovariancePrior):
    def __init__(
        self,
        sigma_y_bounds: Bound = Bound(1e-3, 10),
        sigma_y: Optional[float] = None,
        l: Optional[float] = None,
    ) -> None:
        set_params = {}
        if sigma_y is not None or l is not None:
            assert sigma_y is not None and l is not None, "You must set all parameters or none at all."
            set_params={"l": l, "sigma_y": sigma_y}
        super().__init__(
            {
                "sigma_y": sigma_y_bounds,
                "l": Bound(1e-3, 1),
            },
            set_params,
        )
        self._nugget_const = 1e-13  # To prevent division by zero in case l=0

    def _covariance_mat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        K = np.zeros((x.shape[0], y.shape[0]))
        sigma_y_sqr = self.sigma_y**2
        denominator = 2 * (self.l + self._nugget_const) ** 2
        for i in range(K.shape[0]):
            K[i, :] = sigma_y_sqr * np.exp(
                -np.sum((x[i, :] - y) ** 2, axis=1) / denominator
            )
        return K

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> None:
        def dCov_dSigmaY(x, y):
            K = np.zeros((x.shape[0], y.shape[0]))
            sigmay_2 = 2 * self.sigma_y
            denominator = 2 * (self._params["l"] + self._nugget_const) ** 2
            for i in range(K.shape[0]):
                K[i, :] = sigmay_2 * np.exp(-np.sum((x[i, :] - y) ** 2) / denominator)
            return K

        def dCov_dSigmaN(x, y):
            return self.sigma_n * 2 * np.eye(x.shape[0])

        def dCov_dSigmaL(x, y):
            K = np.zeros((x.shape[0], y.shape[0]))
            sigma_y_sqr = self.sigma_y**2
            denominator = 2 * (self.l + self._nugget_const) ** 2
            for i in range(K.shape[0]):
                residuals_sqr = np.sum((x[i, :] - y) ** 2)
                K[i, :] = (
                    sigma_y_sqr * np.exp(-residuals_sqr / denominator) * residuals_sqr
                ) / (self.l + self._nugget_const) ** 3
            return K

        self._grads = {
            "sigma_y": dCov_dSigmaY(x, x),
            "sigma_n": dCov_dSigmaN(x, x),
            "l": dCov_dSigmaL(x, x),
        }
