# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 7022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Simple Gaussian Process Regression with Numpy.
"""

import tqdm
import pickle
import numpy as np

from typing import Tuple
from numpy.linalg import LinAlgError

from .priors import MeanPrior, CovariancePrior


class GaussianProcess:
    """
    Gaussian Process with fixed mean and covariance functions as priors, and known noise level of
    the training data.

    Input
    -----
    mean_prior: The mean function prior.
    covariance_prior: The covariance function prior.
    use_scipy: Whether to use scipy to avoid computing the inverse of the covariance kernel.
    """

    def __init__(
        self,
        mean_prior: MeanPrior,
        covariance_prior: CovariancePrior,
        use_scipy=False,
    ):
        self._use_scipy = use_scipy
        self._x, self._y = None, None
        self._mu = None  # Mean vector of training cases
        self._K = None  # Covariance matrix of training cases
        self._K_inv = None  # Inverse of K
        self._mean_prior = mean_prior
        self._covariance_prior = covariance_prior

    def mean(self, x: np.ndarray) -> np.ndarray:
        return self._mean_prior(x)

    def covariance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._covariance_prior(x, y)

    def _log_marginal_likelihood(self) -> float:
        """
        Computes the log marginal likelihood of the data given the model (assuming a Gaussian
        distribution for the data). When optimising the prior's hyperparameters, we update the GP's
        covariance matrix and mean vector which allows us to compute this quantity: the probability
        of the data given the hyperparameters.
        """
        assert self._x is not None and self._y is not None, "No training data!"
        n = self._x.shape[0]  # Number of training samples
        # Construct the correlation matrix, add a slight constant for numerical stability (due to
        # the inverstion of K and numerical errors of floating point matrix multiplications) and
        # compute the inverse.
        mu, K = (
            self.mean(self._x),
            self._covariance_prior(
                self._x, self._x, observations=True  # Add the noise variance estimate
            ),
        )
        inv_K = np.linalg.inv(K)

        # Compute the log likelihood
        return (
            -0.5 * np.log(np.linalg.det(K))
            - 0.5 * ((self._y - mu).T @ inv_K @ (self._y - mu)).item()
            - n / 2 * np.log(np.pi * 2)
        )

    def _optimize_model(
        self,
        eps: float,
        max_iterations: int,
        lr: float,
        random_init=True,
        init_iter=0,
        init_logml=float("-inf"),
    ) -> Tuple[float, float]:
        assert self._x is not None and self._y is not None, "No training data!"
        i, logml, last_logml = init_iter, float("-inf"), init_logml
        if random_init:
            self._mean_prior.random_init()
            self._covariance_prior.random_init()
            print(f"\t-> Initial mean params: {self._mean_prior.params_str}")
            print(f"\t-> Initial covariance params: {self._covariance_prior.params_str}")
        with tqdm.tqdm(bar_format="{desc}{postfix}") as pbar:
            while i < max_iterations:
                pbar.set_description(f"Iteration {i}")
                pbar.update()
                try:
                    # First the mean!
                    # Set observations=True to add the noise variance estimate
                    K_inv = np.linalg.inv(
                        self._covariance_prior(self._x, self._x, observations=True)
                    )
                    self._mean_prior.optimize_one_step(
                        {"K_inv": K_inv}, self._x, self._y, lr
                    )
                    # Then the covariance!
                    mu = self.mean(self._x)
                    self._covariance_prior.optimize_one_step(
                        {"mu": mu}, self._x, self._y, lr
                    )
                    # Now compute the total log ML
                    logml = self._log_marginal_likelihood()
                except LinAlgError:
                    break
                pbar.set_postfix_str(f"Log marginal likelihood: {logml:.2f}")
                if np.abs(last_logml - logml) <= eps:
                    break
                last_logml = logml
                i += 1
        return i, logml

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_restarts=10,
        lr=1e-3,
        eps=1e-5,
        max_iterations=1000,
    ):
        """
        Fit the prior mean and covariance functions to the data points via marginal likelihood
        maximization. Having this function take X and Y as arguments forces the user to call it and
        makes the GP class more intuitive to use.
        """
        print(
            f"[*] Fitting the training data with log marginal likelihood maximization..."
        )
        # TODO: Multiprocessing
        self._n_points = x.shape[0]
        self._x, self._y = x, y
        # Optimise prior parameters
        n_restarts = 1 if n_restarts == 0 else n_restarts
        models = {}

        for n in range(n_restarts):
            print(f"[*] Optimizing model {n+1}/{n_restarts}...")
            iteration, logml = self._optimize_model(1e-3, max_iterations, lr)
            models[logml] = {
                "mean": pickle.dumps(self._mean_prior),
                "cov": pickle.dumps(self._covariance_prior),
                "iteration": iteration,
            }
        best_logml = max(models.keys())
        best_model = models[best_logml]
        print(
            f"\n[*] Fine-tuning best model with log marginal likelihood {best_logml}..."
        )
        self._mean_prior = pickle.loads(best_model["mean"])
        self._covariance_prior = pickle.loads(best_model["cov"])
        _, logml = self._optimize_model(
            eps,
            10 * max_iterations,
            lr,
            random_init=False,
            init_iter=best_model["iteration"],
            init_logml=best_logml,
        )
        print(
            "\n\n*****************************************************************************"
        )
        print(f"**Final log marginal likelihood: {logml}")
        print(f"**Optimal mean params: {self._mean_prior.params_str}")
        print(f"**Optimal covariance params: {self._covariance_prior.params_str}")
        print(
            "*****************************************************************************"
        )
        # Now we can set mu, K, and K_inv
        self._mu = self.mean(x)
        self._K = self._covariance_prior(x, x, observations=True)
        if not self._use_scipy:
            self._K_inv = np.linalg.inv(self._K)

    def predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GP model predicting.

        Input
        -----
        x_test: test set, array of shape (n_samples, n_features)

        Output
        ------
        f: GP predictions
        var: Prediction variances
        """
        assert self._x is not None and self._y is not None, "No training data!"
        assert self._K is not None and self._mu is not None, "GP not fitted!"
        # Correlation matrix between the training and test data
        K_train_test = self._covariance_prior(self._x, x_test)
        K_test = self._covariance_prior(x_test, x_test)
        mu_test = self._mean_prior(x_test)

        ############ METHOD 1: Matrix-based with matrix inverse #################
        if not self._use_scipy:
            posterior_mean = mu_test + K_train_test.T @ self._K_inv @ (
                self._y - self._mu
            )
            posterior_K = K_test - (K_train_test.T @ self._K_inv @ K_train_test)
        #########################################################################
        ############ METHOD 2: Matrix-based with linear system solving ##########
        ## Using this method improves the speed and numerical accuracy compared to computing the
        ## inverse of K directly. Especially since it can make use of the fact that K is symmetric
        ## and positive definite. See https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/
        else:
            from scipy import linalg

            solved = linalg.solve(self._K, K_train_test, assume_a="pos").T
            posterior_K = K_test - solved @ K_train_test
            posterior_mean = mu_test + solved @ (self._y - self._mu)
        #########################################################################
        ################# GENERATE SAMPLES #######################################
        # nugget = 1e-9 * np.eye(posterior_K.shape[0])
        # f = posterior_mean + np.linalg.cholesky(
        # posterior_K + nugget
        # ) @ np.random.normal(size=(x_test.shape[0], 1))
        # Or:
        f = np.random.multivariate_normal(
            posterior_mean.flatten(), posterior_K, check_valid="ignore"
        )
        # Take the diagonal to obtain the variance
        return f, posterior_mean.flatten(), np.diag(posterior_K).flatten()
