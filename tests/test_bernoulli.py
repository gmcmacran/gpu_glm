########################################
# Testing IRLS with bernoulli settings
########################################
import numpy as np
import pytest
import scipy.stats as stats
from numpy.testing import assert_equal

from gpu_glm import bernoulli_glm


####################
# helpers to test results
####################
def check_results(model, Beta, X, Y, cutoff=0.1):
    T1 = np.all(model.coef().shape == Beta.shape)
    T2 = np.sum(np.abs(model.coef() - Beta)) < cutoff
    T3 = model.predict_proba(X).min() >= 0 and model.predict_proba(X).max() <= 1
    T4 = model.predict(X).min() == 0 and model.predict(X).max() == 1
    T5 = np.mean(model.predict(X) == Y) > np.mean(Y)
    return T1 and T2 and T3 and T4 and T5


def make_dataset(N, Beta, link):
    np.random.seed(1)

    X = np.concatenate([np.random.normal(size=[N, 2]), np.ones([N, 1])], axis=1)
    eta = np.matmul(X, Beta)

    if link == "logit":

        def inv_link(eta):
            return np.exp(eta) / (1 + np.exp(eta))
    elif link == "probit":

        def inv_link(eta):
            norm = stats.norm
            return norm.cdf(eta)
    else:
        print("invalid link")

    mu = inv_link(eta)
    Y = np.random.binomial(1, mu)

    return X, Y


def test_intercept_fit():
    links = ["logit", "probit"]
    for link in links:
        if link == "logit" or link == "probit":
            Beta = np.array([0.04, 0.02, 0.015])
            X, Y = make_dataset(N=10000, Beta=Beta, link=link)
            cutoff = 1

        model = bernoulli_glm(link)
        model.fit(X, Y)

        assert_equal(check_results(model, Beta, X, Y, cutoff), True)


def test_fit():
    links = ["logit", "probit"]
    for link in links:
        if link == "logit" or link == "probit":
            Beta = np.array([0.04, 0.02, 0.015])
            X, Y = make_dataset(N=10000, Beta=Beta, link=link)
            X = X[:, [0, 1]]  # Remove intercept
            cutoff = 1

        model = bernoulli_glm(link)
        model.fit(X, Y)

        assert_equal(check_results(model, Beta, X, Y, cutoff), True)


def test_valid_links():
    with pytest.raises(ValueError):
        bernoulli_glm("foo")
