from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Tuple

class BaseModel(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns mean and variance predictions for inputs X"""
        pass

    @abstractmethod
    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> torch.Tensor:
        """Returns kernel matrix K(X1, X2)"""
        pass

    @property
    @abstractmethod
    def X_train(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def Y_train(self) -> np.ndarray:
        pass


__all__ = ["BaseModel"]
