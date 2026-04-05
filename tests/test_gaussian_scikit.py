########################################
# Compare gaussian_glm (identity link)
# to scikit-learn Ridge (cholesky solver)
########################################
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import Ridge

from gpu_glm import gaussian_glm


def make_dataset(N, seed=42):
    np.random.seed(seed)
    X = np.random.randn(N, 3)
    true_coef = np.array([2.0, -1.0, 0.5])
    true_intercept = 3.0
    y = X @ true_coef + true_intercept + np.random.randn(N) * 0.5
    return X, y


def fit_and_round(alpha):
    X, y = make_dataset(5000)
    ridge = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    ridge.fit(X, y)
    model = gaussian_glm(link="identity", alpha=alpha)
    model.fit(X, y)
    ridge_all = np.concatenate([ridge.coef_, [ridge.intercept_]])
    ours_all = model.coef()
    return np.round(ridge_all, 2), np.round(ours_all, 2)


def test_alpha_0_matches_ridge():
    ridge_r, ours_r = fit_and_round(0.0)
    assert_allclose(ours_r, ridge_r, atol=0.005)


def test_alpha_01_matches_ridge():
    ridge_r, ours_r = fit_and_round(0.1)
    assert_allclose(ours_r, ridge_r, atol=0.005)


def test_alpha_1_matches_ridge():
    ridge_r, ours_r = fit_and_round(1.0)
    assert_allclose(ours_r, ridge_r, atol=0.005)


def test_alpha_5_matches_ridge():
    ridge_r, ours_r = fit_and_round(5.0)
    assert_allclose(ours_r, ridge_r, atol=0.005)


def test_alpha_10_matches_ridge():
    ridge_r, ours_r = fit_and_round(10.0)
    assert_allclose(ours_r, ridge_r, atol=0.005)
