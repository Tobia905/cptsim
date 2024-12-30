import numpy as np
from numpy.typing import ArrayLike


def progressive_tax(
    income: ArrayLike, 
    min_tax: float, 
    max_tax: float, 
    min_inc: int | float, 
    max_inc: int | float, 
    k: float = 0.001
) -> np.ndarray | float:
    """
    Computes the progressive taxation using a scaled sigmoid
    function.
    """
    mid_inc = (min_inc + max_inc) / 2
    return (
        min_tax + (max_tax - min_tax) 
        / 
        (1 + np.exp(-k * (income - mid_inc)))
    )