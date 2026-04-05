########################################
# Testing L2 regularization (alpha)
########################################
import numpy as np
from numpy.testing import assert_allclose
import pytest
from gpu_glm import gaussian_glm


def make_dataset(N, seed=1):
    np.random.seed(seed)
    X = np.concatenate(
        [np.random.uniform(low=1, high=2, size=[N, 2]), np.ones([N, 1])], axis=1
    )
    Beta = np.array([0.5, 1, 1.5])
    eta = X @ Beta
    mu = eta
    y = np.random.normal(mu, scale=1)
    X = X[:, [0, 1]]
    return X, y, Beta[:2]


def test_alpha_default_is_zero():
    model = gaussian_glm()
    assert model._alpha == 0.0


def test_alpha_shrinks_coefficients():
    X, y, _ = make_dataset(10000)
    model_unreg = gaussian_glm(alpha=0.0)
    model_reg = gaussian_glm(alpha=1.0)
    model_unreg.fit(X, y)
    model_reg.fit(X, y)
    unreg_norm = np.sum(np.abs(model_unreg.coef()[:-1]))
    reg_norm = np.sum(np.abs(model_reg.coef()[:-1]))
    assert reg_norm < unreg_norm


def test_alpha_zero_matches_unregularized():
    X, y, _ = make_dataset(10000)
    model_default = gaussian_glm()
    model_explicit_zero = gaussian_glm(alpha=0.0)
    model_default.fit(X, y)
    model_explicit_zero.fit(X, y)
    assert_allclose(model_default.coef(), model_explicit_zero.coef(), rtol=1e-10)


def test_larger_alpha_more_shrinkage():
    X, y, _ = make_dataset(10000)
    model_low = gaussian_glm(alpha=0.1)
    model_high = gaussian_glm(alpha=10.0)
    model_low.fit(X, y)
    model_high.fit(X, y)
    low_norm = np.sum(np.abs(model_low.coef()[:-1]))
    high_norm = np.sum(np.abs(model_high.coef()[:-1]))
    assert high_norm < low_norm


def test_intercept_not_regularized():
    X, y, _ = make_dataset(10000)
    model_unreg = gaussian_glm(alpha=0.0)
    model_reg = gaussian_glm(alpha=10.0)
    model_unreg.fit(X, y)
    model_reg.fit(X, y)
    unreg_intercept = model_unreg.coef()[-1]
    reg_intercept = model_reg.coef()[-1]
    assert abs(reg_intercept - unreg_intercept) < 0.5

def test_valid_alpha():
    with pytest.raises(ValueError):
        gaussian_glm(alpha = -42.0)
