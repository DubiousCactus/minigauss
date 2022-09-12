# MiniGauss: a numpy-only Gaussian Process library

## Introduction

I built this library to learn about Gaussian Process and their optimization. The code is highly
extensible and adding prior mean/covariance functions is easy as cake! They can be added with
minimal code:

```python

from minigauss.priors  implement CovariancePrior, MeanPrior

class MyCovariancePrior(CovariancePrior):
    def __init__(
        self,
        parameter_bound: Bound = Bound(1e-3, 10),
    ) -> None:
	super().__init__({"parameter": parameter_bound})

    def _covariance_mat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
	K = ...
	return K

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> None:
        self._grads = {...}


class MyMeanPrior(MeanPrior):
    def __init__(self, parameter_bound: Bound = Bound(-20, 20)) -> None:
	super().__init__({"parameter": parameter_bound})

    def __call__(self, x: np.ndarray) -> np.ndarray:
	return self.parameter * x

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> None:
        self._grads = {...}
```


To fit a prior onto training data, several optimization algorithms can be implemented. For now,
gradient ascent is available.

## Installation

Simply run `python setup.py install` and you're reading to go.

### Requirements
```
numpy
tqdm
```
Note that for better numerical stability and efficiency, `scipy` can be used to solve the system of
linear equations to avoid computing the inverse of the covariance matrix[^1] when computing the
posterior, as such: 
```python
gp = GaussianProcess(MyMeanPrior(), MyCovariancePrior(), use_scipy=True)
```

[^1]: https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/  

## Usage

Example usage:
```python
from minigauss import GaussianProcess
from minigauss.priors import ExponentialKernel, PolynomialFunc

def test_function_1D(x):
    return 1 / 4 * x**2

NUM_TRAIN_PTS = 40
NOISE_STD = 0.9
X_RANGE = (-5, 5)

rng = default_rng()
x_train = rng.uniform(X_RANGE[0], X_RANGE[1], (NUM_TRAIN_PTS, 1))
y_train = test_function_1D(x_train)
y_train += NOISE_STD * rng.standard_normal((NUM_TRAIN_PTS, 1))

gp = GaussianProcess(PolynomialFunc(2), ExponentialKernel())
gp.fit(x_train, y_train, n_restarts=10, lr=1e-3)

# GP model predictions
f_sample1, mean, mean_var = gp.predict(x_oracle)
f_sample2, mean, mean_var = gp.predict(x_oracle)
f_sample3, mean, mean_var = gp.predict(x_oracle)
```

Have a look in the examples folder!


## TODO

- [x] Optimization of log marginal likelihood for hyperparameter tuning of priors
- [ ] Optimization algorithms
	- [x] Gradient ascent
	- [ ] Quasi-newton methods
	 - [ ] ...
- [ ] Multiprocessing for faster initial points search
- [ ] Implement kernel functions
	- [x] Exponential
	- [ ] Matern
	- [ ] ...
- [ ] Implement mean functions
	- [x] Polynomial
	- [x] Constant
	- [ ] ...
- [ ] 2D examples
