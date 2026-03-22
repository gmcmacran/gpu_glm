from abc import ABC, ABCMeta, abstractmethod

import numpy as _np

try:
    import cupy as _cp

    CUPY_AVAILABLE = True
except ImportError:
    _cp = None
    CUPY_AVAILABLE = False

import scipy.stats as stats


def xp():
    """Return the active array module (CuPy if available, else NumPy)."""
    return _cp if CUPY_AVAILABLE else _np


def backend_info():
    """
    Return a human‑readable description of the active compute backend.

    Returns
    -------
    str
        A multi‑line string describing whether the package is using
        NumPy (CPU) or CuPy (GPU), and GPU details if available.
    """
    if not CUPY_AVAILABLE:
        return "Backend: NumPy (CPU)\nCuPy not installed. All computations run on CPU."

    # CuPy is available → gather GPU info
    try:
        device_id = _cp.cuda.runtime.getDevice()
        props = _cp.cuda.runtime.getDeviceProperties(device_id)

        name = props["name"].decode("utf-8")
        total_mem = props["totalGlobalMem"] / (1024**3)
        mp_count = props["multiProcessorCount"]

        cuda_rt = _cp.cuda.runtime.runtimeGetVersion()
        cuda_drv = _cp.cuda.runtime.driverGetVersion()

        return (
            "Backend: CuPy (GPU)\n"
            f"Device: {name}\n"
            f"Total Memory: {total_mem:.2f} GB\n"
            f"Multiprocessors: {mp_count}\n"
            f"CUDA Runtime Version: {cuda_rt}\n"
            f"CUDA Driver Version: {cuda_drv}"
        )

    except Exception as e:
        # Fallback if GPU query fails
        return (
            "Backend: CuPy (GPU)\n"
            "CuPy is installed, but GPU properties could not be retrieved.\n"
            f"Error: {e}"
        )


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
        self._B = None
        self._link = link
        super().__init__()

    # -----------------------------
    # Backend helpers
    # -----------------------------
    def _to_backend(self, arr):
        """Convert input to backend array (CuPy if available, else NumPy)."""
        if CUPY_AVAILABLE:
            return _cp.asarray(arr)
        return _np.asarray(arr)

    def _to_numpy(self, arr):
        """Convert backend array to NumPy (for SciPy/stats)."""
        if CUPY_AVAILABLE:
            return _cp.asnumpy(arr)
        return arr

    # -----------------------------
    # Public API
    # -----------------------------
    def coef(self):
        """
        Return the fitted coefficient vector.

        Returns
        -------
        np.ndarray
            The coefficient vector ``B`` of shape ``(n_features,)``.
        """
        return self._to_numpy(self._B)

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
        xp_backend = xp()

        # Convert to backend (CuPy or NumPy)
        X = self._to_backend(X)
        Y = self._to_backend(Y)

        n_features = X.shape[1]

        # Initialize coefficients
        self._B = xp_backend.zeros(n_features)
        self._B[-1] = Y.mean()

        tol = 1e6
        while tol > 1e-5:
            eta = X.dot(self._B)
            mu = self._inv_link(eta)

            # Vectorized weights
            w = (
                1 / (self._var_mu(mu) * self._a_of_phi(Y, mu, self._B))
            ) * xp_backend.power(self._del_eta_del_mu(mu), 2)

            # Vectorized z
            z = (Y - mu) * self._del_eta_del_mu(mu) + eta

            # Weighted least squares without forming diag(W)
            # X^T W X  ==  (X * w[:, None]).T @ X
            Xw = X * w[:, None]
            XtWX = Xw.T.dot(X)
            XtWz = Xw.T.dot(z)

            # Solve for update
            B_new = xp_backend.linalg.solve(XtWX, XtWz)

            tol = xp_backend.sum(xp_backend.abs(B_new - self._B))
            self._B = B_new

        return self

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
        Xb = self._to_backend(X)
        eta = Xb.dot(self._B)
        mu = self._inv_link(eta)
        return self._to_numpy(mu)

    # -----------------------------
    # Link functions
    # -----------------------------
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
        xp_backend = xp()

        if self._link == "identity":
            return eta
        elif self._link == "log":
            return xp_backend.exp(eta)
        elif self._link == "inverse":
            return 1 / eta
        elif self._link == "logit":
            e = xp_backend.exp(eta)
            return e / (1 + e)
        elif self._link == "probit":
            eta_np = self._to_numpy(eta)
            return self._to_backend(stats.norm.cdf(eta_np))
        elif self._link == "sqrt":
            return xp_backend.power(eta, 2)
        elif self._link == "1/mu^2":
            return 1 / xp_backend.power(eta, 0.5)

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
        xp_backend = xp()

        if self._link == "identity":
            return xp_backend.ones(mu.shape)
        elif self._link == "log":
            return 1 / mu
        elif self._link == "inverse":
            return -1 / xp_backend.power(mu, 2)
        elif self._link == "logit":
            return 1 / (mu * (1 - mu))
        elif self._link == "probit":
            mu_np = self._to_numpy(mu)
            return self._to_backend(stats.norm.pdf(stats.norm.ppf(mu_np)))
        elif self._link == "sqrt":
            return 0.5 * xp_backend.power(mu, -0.5)
        elif self._link == "1/mu^2":
            return -2 / xp_backend.power(mu, 3)

    # -----------------------------
    # Abstract variance + dispersion
    # -----------------------------
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


# -----------------------------
# GLM Subclasses
# -----------------------------
class gaussian_glm(IRLS):
    """
    Gaussian GLM with identity, log, or inverse link.
    """

    def __init__(self, link="identity"):
        if link in ("identity", "log", "inverse"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        return xp().ones(mu.shape)

    def _a_of_phi(self, Y, mu, B):
        xp_backend = xp()
        return xp_backend.sum((Y - mu) ** 2) / (Y.shape[0] - B.shape[0])


class bernoulli_glm(IRLS):
    """
    Bernoulli GLM with logit or probit link.
    """

    def __init__(self, link="logit"):
        if link in ("logit", "probit"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        return mu * (1 - mu)

    def _a_of_phi(self, Y, mu, B):
        return xp().ones(Y.shape[0])

    def predict_proba(self, X):
        Xb = self._to_backend(X)
        props = self._inv_link(Xb.dot(self._B))
        props = self._to_numpy(props)
        return _np.column_stack([1 - props, props])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


class poisson_glm(IRLS):
    """
    Poisson GLM with log, identity, or sqrt link.
    """

    def __init__(self, link="log"):
        if link in ("log", "identity", "sqrt"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        return mu

    def _a_of_phi(self, Y, mu, B):
        return xp().ones(Y.shape[0])


class gamma_glm(IRLS):
    """
    Gamma GLM with inverse, identity, or log link.
    """

    def __init__(self, link="inverse"):
        if link in ("inverse", "identity", "log"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        return mu**2

    def _a_of_phi(self, Y, mu, B):
        xp_backend = xp()
        numerator = (Y - mu) ** 2
        denominator = mu**2 * (Y.shape[0] - B.shape[0])
        phi = xp_backend.sum(numerator / denominator)
        return xp_backend.ones(Y.shape[0]) * phi


class inverse_gaussian_glm(IRLS):
    """
    Inverse Gaussian GLM with 1/μ², inverse, identity, or log link.
    """

    def __init__(self, link="1/mu^2"):
        if link in ("1/mu^2", "inverse", "identity", "log"):
            super().__init__(link)
        else:
            raise ValueError(f"Invalid link: {link}")

    def _var_mu(self, mu):
        return mu**3

    def _a_of_phi(self, Y, mu, B):
        xp_backend = xp()
        return -xp_backend.sum((Y - mu) ** 2) / (Y.shape[0] - B.shape[0])
