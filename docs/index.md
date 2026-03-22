# gpu_glm

A lightweight Python implementation of Generalized Linear Models (GLMs) that runs on a GPU. 

This package provides:

- Gaussian, Bernoulli, Poisson, Gamma, and Inverse Gaussian models.
- Multiple link functions (identity, log, inverse, logit, probit, etc.)
- A Cupy-based implementation that falls back to Numpy.
- A sci-kit learn interface.

---

## Installation
To use the GPU, cupy must be installed with a GPU dependancies already working. If cupy is unavailable, numpy is used.  

```bash
pip install gpu_glm
```

A conda package to handle GPU dependancies is under development.


## Quick Example

Below fits a linear regression model.

```python
import numpy as np
from gpu_glm import gaussian_glm
from sklearn.metrics import root_mean_squared_error

# Simulated data
X = np.column_stack([np.random.randn(100), np.ones(100)])
y = 2 * X[:, 0] + 3 +  np.random.randn(100)

# Fit model
model = gaussian_glm()
model.fit(X, y)
print(f"coefficients: {model.coef()}")

y_hat = model.predict(X)
rmse = root_mean_squared_error(y, y_hat)
print(f"RMSE: {np.round(rmse, 3)}")
```