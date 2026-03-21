from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import scipy.stats as stats


class IRLS(ABC):
    """
    Base class implementing the Iteratively Reweighted Least Squares (IRLS)
    algorithm for fitting Generalized Linear Models (GLMs).

    Subclasses must implement:

    - ``_var_mu(mu)``: variance function of the mean
    - ``_a_of_phi(Y, mu, B)``: dispersion-related function

    Parameters
    ----------
    link : str
        Name of the link function. Supported values depend on the subclass.
    """

    __metaclass__ = ABCMeta

    def __init__(self, link):
        """
        Initialize the IRLS model.

        Parameters
        ----------
        link : str
            The link function to use (e.g., ``"identity"``, ``"log"``,
            ``"logit"``, ``"inverse"``, ``"probit"``, ``"sqrt"``,
            ``"1/mu^2"``).
        """
        self._B = np.zeros([0])  # coefficient vector
        self._link = link  # link function name
        super().__init__()

    def coef(self):
        """
        Return the fitted coefficient vector.

        Returns
        -------
        np.ndarray
            The coefficient vector ``B`` of shape ``(n_features,)``.
        """
        return self._B

    def fit(self, X, Y):
        """
        Fit the GLM using the IRLS algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix.
        Y : np.ndarray, shape (n_samples,)
            Response vector.

        """
        self._B = np.zeros([X.shape[1]])
        self._B[X.shape[1] - 1] = np.mean(Y)

        tol = 1000
        while tol > 0.00001:
            eta = X.dot(self._B)
            mu = self._inv_link(eta)

            _w = (1 / (self._var_mu(mu) * self._a_of_phi(Y, mu, self._B))) * np.power(
                self._del_eta_del_mu(mu), 2
            )

            W = np.diag(_w)
            z = (Y - mu) * self._del_eta_del_mu(mu) + eta

            B_update = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(z)

            tol = np.sum(np.abs(B_update - self._B))
            self._B = B_update.copy()

    def _inv_link(self, eta):
        """
        Apply the inverse link function.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor.

        Returns
        -------
        np.ndarray
            Mean response ``mu``.
        """
        if self._link == "identity":
            return eta
        elif self._link == "log":
            return np.exp(eta)
        elif self._link == "inverse":
            return 1 / eta
        elif self._link == "logit":
            return np.exp(eta) / (1 + np.exp(eta))
        elif self._link == "probit":
            return stats.norm.cdf(eta)
        elif self._link == "sqrt":
            return np.power(eta, 2)
        elif self._link == "1/mu^2":
            return 1 / np.power(eta, 1 / 2)

    def _del_eta_del_mu(self, mu):
        """
        Compute derivative :math:`d\\eta/d\\mu` for the link function.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Derivative ``dη/dμ`` evaluated at ``mu``.
        """
        if self._link == "identity":
            return np.ones([mu.shape[0]])
        elif self._link == "log":
            return 1 / mu
        elif self._link == "inverse":
            return -1 / np.power(mu, 2)
        elif self._link == "logit":
            return 1 / (mu * (1 - mu))
        elif self._link == "probit":
            return stats.norm.pdf(stats.norm.ppf(mu))
        elif self._link == "sqrt":
            return 0.5 * np.power(mu, -0.5)
        elif self._link == "1/mu^2":
            return -2 / np.power(mu, 3)

    @abstractmethod
    def _var_mu(self, mu):
        """
        Variance function :math:`\\mathrm{Var}(Y \\mid \\mu)`.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Variance evaluated at ``mu``.
        """
        pass

    @abstractmethod
    def _a_of_phi(self, Y, mu, B):
        """
        Dispersion-related function :math:`a(\\phi)`.

        Parameters
        ----------
        Y : np.ndarray
            Observed response.
        mu : np.ndarray
            Mean response.
        B : np.ndarray
            Coefficient vector.

        Returns
        -------
        np.ndarray or float
            Dispersion-related quantity.
        """
        pass

    def predict(self, X):
        """
        Predict the mean response for new data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        np.ndarray
            Predicted mean response ``mu``.
        """
        return self._inv_link(X.dot(self._B))


class gaussian_glm(IRLS):
    """
    Gaussian GLM with identity, log, or inverse link.
    """

    def __init__(self, link="identity"):
        """
        Initialize a Gaussian GLM.

        Parameters
        ----------
        link : {"identity", "log", "inverse"}, default "identity"
            Link function to use.
        """
        if link in ("identity", "log", "inverse"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        """
        Variance function for Gaussian distribution.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Constant variance of ones.
        """
        return np.ones([mu.shape[0]])

    def _a_of_phi(self, Y, mu, B):
        """
        Estimate dispersion using residual sum of squares.

        Parameters
        ----------
        Y : np.ndarray
            Observed response.
        mu : np.ndarray
            Mean response.
        B : np.ndarray
            Coefficient vector.

        Returns
        -------
        float
            Estimated dispersion.
        """
        return np.sum((Y - mu) ** 2) / (Y.shape[0] - B.shape[0])


class bernoulli_glm(IRLS):
    """
    Bernoulli GLM with logit or probit link.
    """

    def __init__(self, link="logit"):
        """
        Initialize a Bernoulli GLM.

        Parameters
        ----------
        link : {"logit", "probit"}, default "logit"
            Link function to use.
        """
        if link in ("logit", "probit"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        """
        Variance function for Bernoulli distribution.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Variance ``mu * (1 - mu)``.
        """
        return mu * (1 - mu)

    def _a_of_phi(self, Y, mu, B):
        """
        Dispersion for Bernoulli distribution.

        Returns
        -------
        np.ndarray
            Array of ones (dispersion fixed at 1).
        """
        return np.ones([Y.shape[0]])

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, 2)`` with ``P(Y=0)`` and ``P(Y=1)``.
        """
        props = self._inv_link(X.dot(self.coef()))
        return np.column_stack([1 - props, props])

    def predict(self, X):
        """
        Predict class labels (0 or 1).

        Parameters
        ----------
        X : np.ndarray
            Design matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


class poisson_glm(IRLS):
    """
    Poisson GLM with log, identity, or sqrt link.
    """

    def __init__(self, link="log"):
        """
        Initialize a Poisson GLM.

        Parameters
        ----------
        link : {"log", "identity", "sqrt"}, default "log"
            Link function to use.
        """
        if link in ("log", "identity", "sqrt"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        """
        Variance function for Poisson distribution.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Variance equal to ``mu``.
        """
        return mu

    def _a_of_phi(self, Y, mu, B):
        """
        Dispersion for Poisson distribution.

        Returns
        -------
        np.ndarray
            Array of ones (dispersion fixed at 1).
        """
        return np.ones([Y.shape[0]])


class gamma_glm(IRLS):
    """
    Gamma GLM with inverse, identity, or log link.
    """

    def __init__(self, link="inverse"):
        """
        Initialize a Gamma GLM.

        Parameters
        ----------
        link : {"inverse", "identity", "log"}, default "inverse"
            Link function to use.
        """
        if link in ("inverse", "identity", "log"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        """
        Variance function for Gamma distribution.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Variance proportional to ``mu**2``.
        """
        return mu**2

    def _a_of_phi(self, Y, mu, B):
        """
        Method-of-moments estimate of dispersion.

        Parameters
        ----------
        Y : np.ndarray
            Observed response.
        mu : np.ndarray
            Mean response.
        B : np.ndarray
            Coefficient vector.

        Returns
        -------
        np.ndarray
            Estimated dispersion repeated for each sample.
        """
        numerator = (Y - mu) ** 2
        denominator = mu**2 * (Y.shape[0] - B.shape[0])
        phi = np.sum(numerator / denominator)
        return np.ones(Y.shape[0]) * phi


class inverse_gaussian_glm(IRLS):
    """
    Inverse Gaussian GLM with 1/μ², inverse, identity, or log link.
    """

    def __init__(self, link="1/mu^2"):
        """
        Initialize an Inverse Gaussian GLM.

        Parameters
        ----------
        link : {"1/mu^2", "inverse", "identity", "log"}, default "1/mu^2"
            Link function to use.
        """
        if link in ("1/mu^2", "inverse", "identity", "log"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        """
        Variance function for Inverse Gaussian distribution.

        Parameters
        ----------
        mu : np.ndarray
            Mean response.

        Returns
        -------
        np.ndarray
            Variance proportional to ``mu**3``.
        """
        return mu**3

    def _a_of_phi(self, Y, mu, B):
        """
        Negative dispersion estimate (canonical form).

        Parameters
        ----------
        Y : np.ndarray
            Observed response.
        mu : np.ndarray
            Mean response.
        B : np.ndarray
            Coefficient vector.

        Returns
        -------
        float
            Negative dispersion estimate.
        """
        return -np.sum((Y - mu) ** 2) / (Y.shape[0] - B.shape[0])
