"""
Here we will follow a numeration of kernel functions like in the original article:
https://arxiv.org/pdf/1906.10467.pdf
"""
from enum import Enum

import numpy as np


def phi8(x1: float, x2: float) -> (float, float, float):
    """Kernel funcition (8)

    (x1, x2) -> pi * x1 * x2

    :param x1:
    :type x1: float
    :param x2:
    :type x2: float
    :rtype: (float, float, float)
    """
    return (x1, x2, np.pi * x1 * x2)


def phi9(x1: float, x2: float) -> (float, float, float):
    """Kernelt function (9)

    :param x1:
    :type x1: float
    :param x2:
    :type x2: float
    :rtype: (float, float, float)
    """
    return (x1, x2, np.pi * 0.5 * (1 - x1) * (1 - x2))


def phi10(x1: float, x2: float) -> (float, float, float):
    """Kernel function (10)

    :param x1:
    :type x1: float
    :param x2:
    :type x2: float
    :rtype: (float, float, float)
    """
    return (x1, x2, np.exp(x1, x2, (x1 - x2) ** 2 * np.log(np.pi) / 8))


def phi11(x1: float, x2: float) -> (float, float, float):
    """Kernel function (11)

    :param x1:
    :type x1: float
    :param x2:
    :type x2: float
    :rtype: (float, float, float)
    """
    return (x1, x2, np.pi / (3 * np.cos(x1) * np.cos(x2)))


def phi12(x1: float, x2: float) -> (float, float, float):
    """Kernel function (12)

    :param x1:
    :type x1: float
    :param x2:
    :type x2: float
    :rtype: (float, float, float)
    """
    return (x1, x2, np.pi * np.cos(x1) * np.cos(x2))


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization function. Transform input array to [-1.0, 1.0]

    :param x: 1-dimensional NumPy array
    :type x: np.ndarray
    :rtype: np.ndarray
    """
    if len(x.shape) > 1 and x.shape[1] > 1:
        raise ValueError("Array must be 1-dimensional!")

    min_ = x.min()
    max_ = x.max()

    return 2 * (x - min_) / (max_ - min_) - 1.0


class KernelFunctions(Enum):
    phi8 = phi8
    phi9 = phi9
    phi10 = phi10
    phi11 = phi11
    phi12 = phi12
