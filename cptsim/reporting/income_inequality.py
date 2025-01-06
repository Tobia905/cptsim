from typing import Optional, List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import seaborn as sns

from cptsim.utils import CDF


def gini_index(income: ArrayLike) -> float:
    """
    Computes the Gini index for a given income distribution.
    
    Parameters:
        income (ArrayLike): Array of income values.
        
    Returns:
        gini (float): Gini index.
    """
    # Sort incomes in ascending order
    income = np.sort(income)
    
    n = len(income)
    mean_income = np.mean(income)
    
    # Compute the Gini index using the formula
    diff_sum = np.sum(np.abs(income[:, None] - income))  # Pairwise absolute differences
    gini = diff_sum / (2 * n**2 * mean_income)
    
    return gini


def gini_index_from_lorenz(population: ArrayLike, measures: ArrayLike) -> float:
    """
    Compute the Gini index directly from a Lorenz curve.

    Args:
        population (ArrayLike): Sorted population.
        measures (ArrayLike): Cumulated variable of interest.

    Returns:
        gini (float): Gini index computed from a Lorenz curve.
    """
    # Compute the area under the concentration curve using the trapezoidal rule
    area_under_curve = np.trapz(measures, population)
    
    # Compute Gini index
    return 1 - 2 * area_under_curve


def plot_lorenz_curve(
    distr_ct: ArrayLike, 
    distr_pt: Optional[ArrayLike] = None, 
    labels: Optional[List[str]] = None
) -> None:
    """
    Plot the Lorenz curve for one or more income distributions.

    Args:
        distr_ct (ArrayLike): List of income distributions to compare.
        distr_pt (ArrayLike): List of Post Policy income distributions 
        to compare.
        labels (list of str): Labels for each distribution.

    Returns:
        None
    """
    if distr_pt is not None:
        distrs = [distr_ct, distr_pt] 
        labels = [
            "Constant Taxation | Gini Index: {}", 
            "Progressive Taxation | Gini Index: {}"
        ]

    else:
        distrs = [distr_ct]
        labels = ["Gini Index: {}"]

    for i, distribution in enumerate(distrs):
        gini = round(gini_index(distribution), 3)
        sorted_incomes = np.sort(distribution)
        cumulative_income = np.cumsum(sorted_incomes) / np.sum(sorted_incomes)
        cumulative_population = (
            np.arange(1, len(sorted_incomes) + 1) / len(sorted_incomes)
        )

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


def plot_income_cdf(
    income_pre: ArrayLike, 
    income_post: ArrayLike,
    title: str = "Left Tail Adjustment Plot",
    ylabel: str = "%",
    xlabel: str = "Income"
) -> None:

    cdf_ct, ds_ct = CDF(income_pre)
    cdf_pt, ds_pt = CDF(income_post)

    common_ds = np.linspace(
        min(ds_ct.min(), ds_pt.min()), 
        max(ds_ct.max(), ds_pt.max()), 
        500
    )
    cdf_ct_interp = np.interp(common_ds, ds_ct, cdf_ct)
    cdf_pt_interp = np.interp(common_ds, ds_pt, cdf_pt)

    plt.plot(ds_ct, cdf_ct, label="Pre Policy Income", c="k")
    plt.plot(ds_pt, cdf_pt, label="Post Policy Income", c="k", linestyle="--")
    plt.fill_between(
        common_ds, 
        cdf_ct_interp, 
        cdf_pt_interp, 
        color="gray", 
        alpha=0.3, 
        label="Adjustment"
    )
    plt.grid(alpha=.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
