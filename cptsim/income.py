import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
from numpy.typing import ArrayLike


def _rescale_array(
    values: np.ndarray, 
    left_lim: int | float, 
    right_lim: int | float,
) -> np.ndarray:
    """
    Rescales an array of values from the range [old_min, old_max]
    to the range [new_min, new_max].
    
    Parameters:
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

    stretched_data = clipped_data ** heavy_tail_factor

    return _rescale_array(stretched_data, min_income, max_income)


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


def plot_lorenz_curve(distr_ct: ArrayLike, distr_pt: ArrayLike):
    """
    Plot the Lorenz curve for one or more income distributions.

    Args:
        distributions (list of lists or numpy arrays): List of income distributions to compare.
        labels (list of str): Labels for each distribution.

    Returns:
        None
    """

    labels = [
        "Constant Taxation | Gini Index: {}", 
        "Progressive Taxation | Gini Index: {}"
    ]
    for i, distribution in enumerate([distr_ct, distr_pt]):
        gini = round(gini_index(distribution), 3)
        sorted_incomes = np.sort(distribution)
        cumulative_income = np.cumsum(sorted_incomes) / np.sum(sorted_incomes)
        cumulative_population = np.arange(1, len(sorted_incomes) + 1) / len(sorted_incomes)

        plt.plot(
            cumulative_population, 
            cumulative_income, 
            label=labels[i].format(gini)
        )

    # Plot the equality line
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Equality Line")

    plt.title("Lorenz Curve")
    plt.xlabel("Cumulative Population")
    plt.ylabel("Cumulative Income")
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()


def plot_pre_post_introduction_incomes(distr_ct: ArrayLike, distr_pt: ArrayLike) -> None:
    distr_ct, distr_pt = pd.Series(distr_ct), pd.Series(distr_pt)

    sns.kdeplot(distr_ct, linestyle="--", c="k", zorder=4)
    sns.kdeplot(distr_pt, c="k", zorder=5)

    min_, max_ = (
        min(distr_ct.min(), distr_pt.min()), 
        max(distr_ct.max(), distr_pt.max())
    )
    distr_ct.hist(
        bins=np.linspace(min_, max_, 50), 
        grid=False, 
        edgecolor="k", 
        alpha=.6, 
        density=True, 
        zorder=2, 
        label="Pre Introduction"
    )
    distr_pt.hist(
        bins=np.linspace(min_, max_, 50), 
        grid=False, 
        edgecolor="k", 
        alpha=.6, 
        density=True, 
        zorder=3, 
        label="Post Introduction"
    )
    plt.grid(alpha=.3, zorder=-2)
    plt.title("Pre & Post Progressive Taxation Incomes Distribution")
    plt.legend()
    plt.show()
