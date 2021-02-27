from typing import Callable, List

import pennylane as qml

from .phi_functions import *


def get_feature_map(
    device: qml.Device,
    kernel: KernelFunctions,
    observables: List[qml.operation.Observable],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Generate Feature Map function. Return a function of two NumPy arrays
    that return expected value of observable for each point.

    :param device: Device with 2 wires
    :type device: qml.Device
    :param kernel: Kernel function from list of possible variants
    :type kernel: KernelFunctions
    :param observables: List of obervables. Result of the map will be a Observable[0](0) @ Observable[1](1)
    :type observables: List[qml.operation.Observable]
    :rtype: CCallable[[np.ndarray, np.ndarray], np.ndarray]
    """
    if len(observables) != 2:
        raise ValueError(
            "observables shoud be list of length 2 with observable for each wire!"
        )

    @qml.qnode(device)
    def feature_map(x1, x2, x3):
        """QNode.

        :param x1:
        :param x2:
        :param x3:
        """
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        qml.U1(x1, wires=0)
        qml.U1(x2, wires=1)

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        qml.CNOT(wires=[0, 1])
        qml.U1(x3, wires=1)
        qml.CNOT(wires=[0, 1])

        return qml.expval(observables[0](0) @ observables[1](1))

    def transform_func(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Tranformation function.

        :param x1:
        :type x1: np.ndarray
        :param x2:
        :type x2: np.ndarray
        :rtype: np.ndarray
        """
        x1_ = normalize(x1)
        x2_ = normalize(x2)

        return np.array([feature_map(*kernel(a, b)) for a, b in zip(x1_, x2_)])

    return transform_func
