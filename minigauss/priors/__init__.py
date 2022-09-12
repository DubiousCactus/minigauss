# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Prior abstract class and utilities.
"""

import abc
import numpy as np

from typing import Any, Dict
from dataclasses import dataclass
from numpy.random import default_rng


@dataclass
class Bound:
    # TODO: Make this a lot more flexible.
    min: float
    max: float


class Prior(metaclass=abc.ABCMeta):
    def __init__(self, bounded_params: Dict[str, Bound]) -> None:
        """
        Construct a prior class.

        Input
        -----
        bounded_params: Dictionary of parameter names and their associated default boundaries (for
                        hyperparameter optimization).
        """
        self._default_bounds = {}
        self._params: Dict[str, float] = self._init_params(bounded_params)
        self._grads = {}

    def _init_params(self, params: Dict[str, Bound]) -> Dict[str, float]:
        _params: Dict[str, float] = {}
        rng = default_rng()
        for param, default_bound in params.items():
            self._default_bounds[param] = default_bound
            _params[param] = rng.uniform(
                low=default_bound.min, high=default_bound.max, size=1
            ).item()
            # _params[param] = abs(val) if default_bound.abs_val else val
        return _params

    def random_init(self, bounds: Dict[str, Bound] = {}, verbose=False):
        rng = default_rng()
        for param in self._params.keys():
            if param in bounds:
                bound = bounds[param]
            else:
                bound = self._default_bounds[param]
            # val = rng.normal(scale=bound.std, size=1).item()
            self._params[param] = rng.uniform(
                low=bound.min, high=bound.max, size=1
            ).item()
            # self._params[param] = abs(val) if bound.abs_val else val
        if verbose:
            print(f"[*] Initial values: {self.params_str}")

    @abc.abstractmethod
    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def optimize_one_step(
        self,
        ctx: dict,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        optimizer="gradient_ascent",
    ) -> None:
        """
        Run one step of maximization for the log marginal likelihood via the given optimizer.

        Input
        -----
        ctx: Dictionary context to pass variables to each prior.
        x: Array of inputs.
        y: Array of targets.
        lr: Learning rate.
        optimizer: One of ['gradient_descent', 'L-BFGS-B', 'conjugate_gradient']
        """
        pass

    @property
    def params(self) -> Dict[str, float]:
        return self._params

    @property
    def params_str(self) -> str:
        return ",".join([f"{k}={v:.2f}" for k, v in self._params.items()])

    def __getattribute__(self, __name: str) -> Any:
        if __name in self._params:
            return self._params[__name]
        else:
            raise AttributeError()


from .mean import *
from .covariance import *
