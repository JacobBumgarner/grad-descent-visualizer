"""Example test functions for landscape and descent visualizations."""
from typing import Union

import autograd.numpy as np


def himmelblau_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Himmelblau's function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    return z


def beale_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Beale function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z = (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )
    return z


def sphere_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Sphere function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z = x**2 + y**2
    return z


def matyas_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Matyas function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def three_camel_hump(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Three Camel Hump function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z = 2 * x**2 - 1.05 * x**4 + (x**6 / 6) + x * y + y**2
    return z


def griewank_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Griewank function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z_a = (x**2 + y**2) / 4000
    z_b = np.cos(x / np.sqrt(1)) + np.cos(y / np.sqrt(2))
    z = z_a - z_b + 1
    return z


def bohachevsky_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Bohachevsky function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z = (
        x**2
        + 2 * y**2
        - 0.3 * np.cos(3 * np.pi * x)
        - 0.4 * np.cos(4 * np.pi * y)
        + 0.7
    )
    return z


def zakharov_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Zakharov function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z_a = x**2 + y**2
    z_b = (0.5 * 1 * x + 0.5 * 2 * y) ** 2
    z_c = (0.5 * 1 * x + 0.5 * 2 * y) ** 4
    z = z_a + z_b + z_c
    return z


def easom_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Easom function.

    Recommended range: [-4, 10, -4, 10]

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    exp_pow = -1 * (x - np.pi) ** 2 - (y - np.pi) ** 2
    z = -1 * np.cos(x) * np.cos(y) * np.exp(exp_pow)
    return z


def six_hump_camel_function(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Six hump camel function.

    Args:
        x (Union[float, np.ndarray]):A float or np.ndarray of floats.
        y (Union[float, np.ndarray]): A float or np.ndarray of floats.

    Returns:
        Union[float, np.ndarray]: A float or np.ndarray of floats.
    """
    z_a = (4 - 2.1 * x**2 + x**4 / 3) * x**2
    z_b = x * y
    z_c = (-4 + 4 * y**2) * y**2
    z = z_a + z_b + z_c
    return z
