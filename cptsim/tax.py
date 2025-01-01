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


def redistribute_funds(incomes: ArrayLike, total_funds: float, decay_rate: float) -> np.ndarray:
    """
    Redistribute funds among individuals based on their income using 
    exponential decay.

    Args:
        incomes (ArrayLike): The income levels of individuals.
        total_funds (float): The total amount of money to be distributed.
        decay_rate (float): The rate of exponential decay. Higher values 
        prioritize lower incomes.

    Returns:
        numpy array: The redistributed funds for each individual.
    """
    incomes = np.array(incomes)

    # Apply exponential decay to compute weights
    weights = np.exp(-decay_rate * incomes)

    # Normalize weights to ensure they sum to 1
    normalized_weights = weights / np.sum(weights)

    # Distribute funds based on normalized weights
    redistributed_funds = total_funds * normalized_weights

    return redistributed_funds
