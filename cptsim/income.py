import numpy as np
from scipy.stats import gamma
from numpy.typing import ArrayLike


def simulate_income(
    n: int = 1000, 
    min_income: int | float = 500, 
    max_income: int | float = 7500, 
    median: int | float = 1850
) -> np.ndarray:
    """
    Simulates a left-skewed distribution for post-taxation monthly income
    with narrower density and less spread.
    
    Parameters:
        n (int): Number of samples to generate.
        min_income (float): Minimum income value.
        max_income (float): Maximum income value.
        median (float): Target median income value.
        
    Returns:
        np.ndarray: Array of simulated incomes.
    """
    shape = 3  # Controls skewness
    scale = (median - min_income) / shape  # Adjust scale based on the median
    loc = min_income  # Minimum income corresponds to the location parameter

    # Generate raw Gamma-distributed values
    raw_data = gamma.rvs(shape, loc=loc, scale=scale, size=n)

    # Clip to the desired range to ensure values stay within bounds
    clipped_data = np.clip(raw_data, min_income, max_income)

    return clipped_data


def gini_index(income: ArrayLike) -> float:
    """
    Computes the Gini index for a given income distribution.
    
    Parameters:
        income (np.ndarray): Array of income values.
        
    Returns:
        float: Gini index (0 = perfect equality, 1 = perfect inequality).
    """
    # Sort incomes in ascending order
    income = np.sort(income)
    
    n = len(income)
    mean_income = np.mean(income)
    
    # Compute the Gini index using the formula
    diff_sum = np.sum(np.abs(income[:, None] - income))  # Pairwise absolute differences
    gini = diff_sum / (2 * n**2 * mean_income)
    
    return gini
