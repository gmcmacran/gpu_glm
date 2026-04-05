########################################
# Testing L2 regularization (alpha)
########################################
import numpy as np
import pytest

from gpu_glm import inverse_gaussian_glm


def make_dataset(N, seed=1):
    np.random.seed(seed)
    X = np.concatenate(
        [np.random.uniform(low=1, high=2, size=[N, 2]), np.ones([N, 1])], axis=1
    )
    Beta = np.array([0.5, 1, 1.5])
    eta = X @ Beta
    mu = 1 / np.sqrt(eta)
    y = np.random.wald(mu, 1)
    X = X[:, [0, 1]]
    return X, y, Beta[:2]


def test_alpha_default_is_zero():
    model = inverse_gaussian_glm()
    assert model._alpha == 0.0


def test_alpha_shrinks_coefficients():
    X, y, _ = make_dataset(10000)
    model_unreg = inverse_gaussian_glm(alpha=0.0)
    model_reg = inverse_gaussian_glm(alpha=100.0)
    model_unreg.fit(X, y)
    model_reg.fit(X, y)
    # Coefficients should differ when alpha > 0
    assert not np.allclose(model_unreg.coef(), model_reg.coef(), rtol=1e-6)


def test_alpha_zero_matches_unregularized():
    X, y, _ = make_dataset(10000)
    model_default = inverse_gaussian_glm()
    model_explicit_zero = inverse_gaussian_glm(alpha=0.0)
    model_default.fit(X, y)
    model_explicit_zero.fit(X, y)
    assert np.allclose(model_default.coef(), model_explicit_zero.coef(), rtol=1e-6)


def test_intercept_not_regularized():
    X, y, _ = make_dataset(10000)
    model_unreg = inverse_gaussian_glm(alpha=0.0)
    model_reg = inverse_gaussian_glm(alpha=100.0)
    model_unreg.fit(X, y)
    model_reg.fit(X, y)
    unreg_intercept = model_unreg.coef()[-1]
    reg_intercept = model_reg.coef()[-1]
    assert abs(reg_intercept - unreg_intercept) < 0.1


def test_valid_alpha():
    with pytest.raises(ValueError):
        inverse_gaussian_glm(alpha=-42.0)
