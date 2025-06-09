"""
Core abstractions for the *Modeling* stage of ALPine FOREst.

A Model consumes *X* (frame descriptors) and *Y* (targets from the Evaluator),
fits its internal parameters, and can make probabilistic predictions.

Typical concrete classes wrap GPyTorch GPs, scikit‑learn regressors, etc.
Concrete implementations belong in ``alpfore.models.*``.
"""

from __future__ import annotations

import abc
from typing import Tuple, TYPE_CHECKING
from typing_extensions import Protocol

if TYPE_CHECKING:
    import numpy as np


class BaseModel(abc.ABC):
    """Abstract contract for surrogate models used in active learning."""

    # --------------------------- Training --------------------------- #
    @abc.abstractmethod
    def fit(self, X: "np.ndarray", Y: "np.ndarray") -> None:
        """
        Learn model parameters from data.

        X : shape (n_samples, d)
        Y : shape (n_samples, k)
        """
        ...

    # ------------------------- Prediction -------------------------- #
    @abc.abstractmethod
    def predict(self, X: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Predict mean and variance for new inputs.

        Returns
        -------
        mean : shape (n_samples, k)
        var  : shape (n_samples, k)
        """
        ...


__all__ = ["BaseModel"]

