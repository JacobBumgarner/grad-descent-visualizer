"""The auto-grad and descent functions for computing gradient descent."""
from typing import Callable, Sequence, Union

import autograd.numpy as np
from autograd import elementwise_grad, grad


def prep_array_for_descent(variables: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Prepare an array of XY points for a gradient descent.

    Args:
        variables (Union[Sequence[float], np.ndarray]): The XY points.

    Returns:
        np.ndarray: A 2D array of XY points arranged row-wise.
    """
    # Make sure the points are an array
    if not isinstance(variables, np.ndarray):
        variables = np.array(variables, float)

    # Make sure the points are 2D
    if variables.ndim != 2:
        variables = np.expand_dims(variables, -1)

    if variables.ndim != 2:
        raise ValueError(
            """A 1D or 2D array of XY points must be passed into this function."""
        )

    # Check to make sure that the variables are arranged row-wise
    if np.unique(variables.shape).shape[0] > 1:
        if variables.shape[0] != 2:
            variables = variables.transpose()

    return variables.astype(float)


def save_descent_snapshot(variables: np.ndarray, equation: Callable) -> np.ndarray:
    """Generate an array that stores the snapshot of the variables and their costs.

    Args:
        variables (np.ndarray): The 2D array of the variables.
        equation (Callable): The function being used for the descent.

    Returns:
        np.ndarray: A 3D array of the variable snapshot, arranged row-wise.
    """
    snapshot = np.zeros((3, variables.shape[-1]))  # create an empty array with Z slot
    snapshot[:2, :] = variables
    snapshot[2, :] = equation(*variables)
    return snapshot


def differentiate(
    equation: Callable, element_wise: bool = False
) -> Union[Callable, Sequence[Callable]]:
    """Differentiate the specified function.

    Args:
        equation (Callable): The function that represents an equation
        element_wise (bool): Indicates whether to return element_wise differentials.
            Defaults to False.

    Returns:
        Union[Callable, Sequence[Callable]]: A list of the differential function(s).
    """
    function_parameters = equation.__code__.co_argcount

    if element_wise is False:
        differentials = [grad(equation, index) for index in range(function_parameters)]
    else:
        differentials = [
            elementwise_grad(equation, index) for index in range(function_parameters)
        ]

    return differentials


def compute_gradients(
    differentials: Sequence[Callable], variables: np.ndarray
) -> np.ndarray:
    """Return the computed gradient of a set of points based on input differentials.

    The variables for the in put function must

    Args:
        gradients (Sequence[Callable]): The differnetial equations.
        variables (np.ndarray): The variables for the gradient computation. If
            variables is a 2D array, the variables should be oriented row-wise, not
            column-wise i.e., where the ``variables.shape == (2, n)``

    Returns:
        Union[Sequence[float], np.ndarray]
            Either a list of floats that matches the input length of the equation or
            an np.ndarray.
    """
    if len(differentials) != variables.shape[0]:
        raise ValueError(
            "The number of differential functions must match the number of input "
            "variables."
        )

    gradients = []
    for diff in differentials:
        gradients.append(diff(*variables))

    return np.array(gradients)


def descend_gradient(
    differentials: Sequence[Callable],
    variables: np.ndarray,
    alpha: float = 0.01,
) -> tuple[Sequence[float], Sequence[float]]:
    """Update the variables with the input differential functions.

    Args:
        differentials (Sequence[Callable]): A list or tuple of differential functions
            created using ``differentiate``.
        variables (np.ndarray): A 2D or 3D array of variables that will be updated.
        alpha (float, optional): The learning rate of the descent. Defaults to 0.01.

    Returns:
        Sequence[Sequence[float], Sequence[float]]: The list of updated variables and
             list of their previous gradients.
    """
    gradients = compute_gradients(differentials, variables)  # find the gradients
    variables -= gradients * alpha

    return variables
