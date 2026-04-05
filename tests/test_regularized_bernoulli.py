########################################
# Testing L2 regularization (alpha)
########################################
import numpy as np
import pytest
from numpy.testing import assert_allclose

from gpu_glm import bernoulli_glm


def make_dataset(N, seed=1):
    np.random.seed(seed)
    X = np.concatenate([np.random.normal(size=[N, 2]), np.ones([N, 1])], axis=1)
    Beta = np.array([0.04, 0.02, 0.015])
    eta = X @ Beta
    mu = np.exp(eta) / (1 + np.exp(eta))
    y = np.random.binomial(1, mu)
    X = X[:, [0, 1]]
    return X, y, Beta[:2]


def test_alpha_default_is_zero():
    model = bernoulli_glm()
    assert model._alpha == 0.0


def test_alpha_shrinks_coefficients():
    X, y, _ = make_dataset(10000)
    model_unreg = bernoulli_glm(alpha=0.0)
    model_reg = bernoulli_glm(alpha=1.0)
    model_unreg.fit(X, y)
    model_reg.fit(X, y)
    unreg_norm = np.sum(np.abs(model_unreg.coef()[:-1]))
    reg_norm = np.sum(np.abs(model_reg.coef()[:-1]))
    assert reg_norm < unreg_norm


def test_alpha_zero_matches_unregularized():
    X, y, _ = make_dataset(10000)
    model_default = bernoulli_glm()
    model_explicit_zero = bernoulli_glm(alpha=0.0)
    model_default.fit(X, y)
    model_explicit_zero.fit(X, y)
    assert_allclose(model_default.coef(), model_explicit_zero.coef(), rtol=1e-10)


def test_intercept_not_regularized():
    X, y, _ = make_dataset(10000)
    model_unreg = bernoulli_glm(alpha=0.0)
    model_reg = bernoulli_glm(alpha=10.0)
    model_unreg.fit(X, y)
    model_reg.fit(X, y)
    unreg_intercept = model_unreg.coef()[-1]
    reg_intercept = model_reg.coef()[-1]
    assert abs(reg_intercept - unreg_intercept) < 0.1


def test_valid_alpha():
    with pytest.raises(ValueError):
        bernoulli_glm(alpha=-42.0)
