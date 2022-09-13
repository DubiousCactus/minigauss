# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Mean priors.
"""

import abc
import numpy as np

from typing import Dict, List

from . import Prior, Bound


class MeanPrior(Prior, metaclass=abc.ABCMeta):
    def __init__(self, bounded_params: Dict[str, Bound]) -> None:
        super().__init__(bounded_params)

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    def optimize_one_step(
        self,
        ctx: dict,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        optimizer="gradient_ascent",
    ) -> None:
        assert "K_inv" in ctx, "K_inv not set in context!"
        mu = self(x)
        self._compute_gradients(x, y)
        if optimizer == "gradient_ascent":
            residuals_Kinv = (y - mu).T @ ctx["K_inv"]
            for param in self._params.keys():
                self._params[param] += lr * (residuals_Kinv @ self._grads[param]).item()
        else:
            # TODO: More methods! https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/
            raise NotImplementedError(f"Optimizer {optimizer} is not implemented.")


class PolynomialFunc(MeanPrior):
    def __init__(self, degree: int, bounds: List[Bound] = []) -> None:
        if bounds != []:
            assert len(bounds) == (degree + 1), "You must specify N bounds for a polynomial of degree N"
        super().__init__(
            {
                str(chr(i + ord("a"))): bounds[i] if bounds != [] else Bound(-20, 20)
                for i in range(degree + 1)
            }
        )
        self._degree = degree

    def __call__(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros_like(x)
        for i in range(self._degree + 1):
            res += self.params[str(chr(self._degree - i + ord("a")))] * np.power(x, i)
        return res

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Compute the partial derivatives of the function w.r.t. each parameter.
        This is automated as a function of the degree of the polynomial.
        """
        # Only compute them once, as they're only a function of the input
        if self._grads != {}:
            return
        for i in range(self._degree + 1):
            param = str(chr(self._degree - i + ord("a")))
            self._grads[param] = np.ones_like(x) if i == 0 else np.power(x, i)


class ConstantFunc(MeanPrior):
    def __init__(self, bounds: Bound = Bound(-20, 20)) -> None:
        super().__init__({"c": bounds})

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.c * np.ones_like(x)

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Compute the partial derivatives of the function w.r.t. each parameter.
        This is automated as a function of the degree of the polynomial.
        """
        # Only compute them once, as they're only a function of the input
        if self._grads != {}:
            return
        self._grads["c"] = np.ones_like(x)
