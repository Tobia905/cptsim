import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
from numpy.typing import ArrayLike

from cptsim.reporting.income_inequality import gini_index


def _rescale_array(
    values: np.ndarray, 
    left_lim: int | float, 
    right_lim: int | float,
) -> np.ndarray:
    """
    Rescales an array of values from the range [old_min, old_max]
    to the range [new_min, new_max].
    
    Args:
        values (np.ndarray): Array of values to rescale.
        old_min (float): Minimum value of the old range.
        old_max (float): Maximum value of the old range.
        new_min (float): Minimum value of the new range.
        new_max (float): Maximum value of the new range.
    
    Returns:
        np.ndarray: Rescaled array of values.
    """
    min_, max_ = values.min(), values.max()
    return left_lim + (values - min_) * (right_lim - left_lim) / (max_ - min_)


def simulate_income(
    n: int = 1000, 
    min_income: int | float = 500, 
    max_income: int | float = 15000, 
    median: int | float = 1850,
    heavy_tail_factor: float = 1.2
) -> np.ndarray:
    """
    Simulates a left-skewed distribution for post-taxation monthly income
    with narrower density and less spread.
    
    Args:
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

    stretched_data = clipped_data ** heavy_tail_factor

    return _rescale_array(stretched_data, min_income, max_income)


def plot_income_distribution(
    incomes: ArrayLike, 
    title: str = "Simulated Post-Taxation Monthly Income - Gini Index:"
) -> None:
    sns.kdeplot(incomes, c="r", label="Kernel Density Estimation")
    plt.grid(alpha=.3)
    plt.hist(
        incomes, 
        bins=50, 
        density=True, 
        color='gray',
        alpha=0.7, 
        edgecolor='black'
    )
    plt.title(
        title + f" {gini_index(incomes):.2f}"
    )
    plt.xlabel('Income ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
