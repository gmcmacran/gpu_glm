# gpu_glm

A lightweight Python implementation of Generalized Linear Models (GLMs) using the
Iteratively Reweighted Least Squares (IRLS) algorithm.

This package provides:

- Gaussian, Bernoulli, Poisson, Gamma, and Inverse Gaussian GLMs
- Multiple link functions (identity, log, inverse, logit, probit, etc.)
- A clean, extensible IRLS base class
- NumPy-based implementation with no heavy dependencies

---

## Installation

```bash
pip install gpu_glm
```


## Quick Example

Here’s a minimal working example showing how to fit a Gaussian GLM using the
IRLS algorithm implemented in `gpu_glm`:

```python
import numpy as np
from gpu_glm import gaussian_glm

# Simulated data
X = np.column_stack([np.ones(100), np.random.randn(100)])
Y = 3 + 2 * X[:, 1] + np.random.randn(100)

# Fit model
model = gaussian_glm()
model.fit(X, Y)

print(model.coef())
```
