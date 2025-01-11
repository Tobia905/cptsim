from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.integrate import simpson
from numpy.typing import ArrayLike

from cptsim.agents.economic_agent import EconomicAgent
from cptsim.reporting.income_inequality import gini_index_from_lorenz
from cptsim.utils import prettify_pandas_bins


def _compute_ranked_data(
    income: ArrayLike, goods_bought: ArrayLike, method: str = "average"
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        rankdata(income, method=method), 
        rankdata(goods_bought, method=method) 
    )


def agents_to_df(
    agents_ct: List[EconomicAgent], 
    agents_pt: List[EconomicAgent]
) -> pd.DataFrame:
    
    goods_pt, goods_ct, incomes_pt, incomes_ct = (
        [ag.bought_goods for ag in agents_pt], 
        [ag.bought_goods for ag in agents_ct],
        [ag.income for ag in agents_pt],
        [ag.income for ag in agents_ct]
    )
    income_goods = pd.DataFrame(
        {
            "income_ct": incomes_ct,
            "income_pt": incomes_pt,
            "goods_ct": goods_ct,
            "goods_pt": goods_pt
        }
    )
    return income_goods


def get_cons_ineq_data(
    agents_ct: List[EconomicAgent], 
    agents_pt: List[EconomicAgent]
) -> pd.DataFrame:
    """
    Takes a list of economic agents with constant and progressive
    taxations and turns it into a dataframe with binned versions
    of the income distributions and the relative number of bought
    goods for each bin.

    Args:
        agents_ct (list): economic agents with constant taxation.
        agents_pt (list): economic agents with progressive taxation.

    Returns:
        tot_goods_by_inc (pd.DataFrame): dataframe with binned incomes 
        and bought goods.
    """
    income_goods = agents_to_df(agents_ct, agents_pt)
    income_goods["income_pt_bin"] = pd.cut(income_goods["income_pt"], 20)
    income_goods["income_ct_bin"] = pd.cut(income_goods["income_ct"], 20)
    goods_by_inc_pt = (
        income_goods
        .groupby("income_pt_bin", observed=True)
        .agg({"goods_pt": "sum"})
        .reset_index()
        .rename({"income_pt_bin": "income_bin"}, axis=1)
    )
    goods_by_inc_pt["sample_size_pt"] = (
        income_goods
        .groupby("income_pt_bin", observed=True)
        .size()
        .reset_index()[0]
    )

    goods_by_inc_ct = (
        income_goods
        .groupby("income_ct_bin", observed=True)
        .agg({"goods_ct": "sum"})
        .reset_index()
        .rename({"income_ct_bin": "income_bin"}, axis=1)
    )
    goods_by_inc_ct["sample_size_ct"] = (
        income_goods
        .groupby("income_ct_bin", observed=True)
        .size()
        .reset_index()[0]
    )

    tot_goods_by_inc = pd.merge(
        goods_by_inc_pt, goods_by_inc_ct, on="income_bin", how="outer"
    )
    tot_goods_by_inc["av_goods_pt"] = (
        tot_goods_by_inc["goods_pt"] / tot_goods_by_inc["sample_size_pt"]
    )
    tot_goods_by_inc["av_goods_ct"] = (
        tot_goods_by_inc["goods_ct"] / tot_goods_by_inc["sample_size_ct"]
    )
    tot_goods_by_inc["income_bin"] = prettify_pandas_bins(tot_goods_by_inc["income_bin"])

    return tot_goods_by_inc


def area_between_curve(
    income: ArrayLike, 
    goods_bought: ArrayLike, 
    ranking_method: str = "average"
) -> float:
    """
    Computes the area between the rank vs rank curve and the 
    diagonal (perfect equality line).

    Args:
        income (ArrayLike): Array of individual incomes.
        goods_bought (ArrayLike): Array of goods bought by individuals.
        ranking_method (str): the ranking method in the rankdata function.

    Returns:
        float: Area between the curve and the diagonal.
    """
    # Compute ranks
    income_rank, goods_rank = (
        _compute_ranked_data(income, goods_bought, method=ranking_method)
    )
    
    # Sort by income rank to create the cumulative curve
    sorted_indices = np.argsort(income_rank)
    sorted_income_rank = income_rank[sorted_indices]
    sorted_goods_rank = goods_rank[sorted_indices]
    
    # Normalize ranks to [0, 1]
    sorted_income_rank /= sorted_income_rank.max()
    sorted_goods_rank /= sorted_goods_rank.max()
    
    # Compute the difference from the diagonal
    signed_diff = sorted_goods_rank - sorted_income_rank

    positive_area = simpson(np.maximum(signed_diff, 0), x=sorted_income_rank)
    negative_area = simpson(np.maximum(-signed_diff, 0), x=sorted_income_rank)
    
    # Net signed area
    net_area = positive_area - negative_area
    
    return net_area


def rank_vs_rank_plot(
    pre_tax: ArrayLike, 
    post_tax: ArrayLike, 
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ranking_method: str = "average",
    xlabel: str = "Rank by Income", 
    ylabel: str = "Rank by Goods Bought", 
    title: str = "Rank vs Rank Plot"
) -> None:
    """
    Creates a rank vs rank scatterplot to visualize the concentration 
    of goods bought in higher income levels.

    Args:
        income (ArrayLike): Array of individual incomes.
        goods_bought (ArrayLike): Array of goods bought by individuals.
        ax (plt.Axes): the axes object from matplotlib.
        figsize (tuple): the size of the figure.
        ranking_method (str): the method to rank the data.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    income_rank, goods_rank = (
        _compute_ranked_data(pre_tax, post_tax, method=ranking_method)
    )
    abc = area_between_curve(pre_tax, post_tax)
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(income_rank, goods_rank, color='blue', zorder=2)
    
    # Add a diagonal line (reference for equality)
    max_rank = max(max(income_rank), max(goods_rank))
    ax.plot(
        [0, max_rank], 
        [0, max_rank], 
        color='red', 
        linestyle='--', 
        label="Equality Line", 
        zorder=3
    )
    
    # Add labels and legend
    _ = ax.set_xlabel(xlabel)
    _ = ax.set_ylabel(ylabel)
    _ = ax.set_title(title + f" | ABC: {abc:.3f}")
    _ = ax.legend()
    _ = ax.grid(alpha=0.3, zorder=-2)


def concentration_curve(income: ArrayLike, goods: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the concentration curve for goods bought across income levels.

    Args:
        income (array-like): Income values for individuals.
        goods (array-like): Number of goods bought by individuals.

    Returns:
        cum_population (np.ndarray): Cumulative share of the population 
        (ranked by income).
        cum_goods (np.ndarray): Cumulative share of goods bought.
    """
    # Create a DataFrame to sort both variables by income
    data = pd.DataFrame({"income": income, "goods": goods})
    data = data.sort_values(by="income")
    
    # Compute cumulative sums for goods and normalize
    cum_goods = np.cumsum(data["goods"]) / data["goods"].sum()
    
    # Compute cumulative population proportion
    n = len(data)
    cum_population = np.arange(1, n + 1) / n
    
    return cum_population, cum_goods


def plot_concentration_curve(
    income: ArrayLike, 
    goods: ArrayLike, 
    income_post: Optional[ArrayLike] = None,
    goods_post: Optional[ArrayLike] = None
) -> None:
    """
    Plot the concentration curve and display interpretation.

    Args:
        income (ArrayLike): Income values for individuals.
        goods (ArrayLike): Number of goods bought by individuals.
        income_post (ArrayLike): Optional array of post policy income.
        goods_post (ArrayLike): Optional array of post policy bought goods.

    Returns:
        None
    """
    if income_post is not None and goods_post is None:
        raise ValueError(
            "you need to pass the bought goods array to "
            "compute the concentration curve."
        )
    
    elif goods_post is not None and income_post is None:
        raise ValueError(
            "you need to pass the income array to "
            "sort the bought goods by income."
        )
    
    cum_population, cum_goods = concentration_curve(income, goods)
    gini = gini_index_from_lorenz(cum_population, cum_goods)
    
    if income_post is not None:
        cum_population_pt, cum_goods_pt = concentration_curve(income_post, goods_post)
        gini_pt = gini_index_from_lorenz(cum_population_pt, cum_goods_pt)
        label_pt, label_ct = (
            f"Progressive Taxation | Gini Index: {gini_pt:.3f}", 
            f"Constant Taxation | Gini Index: {gini:.3f}"
        )
        plt.plot(cum_population_pt, cum_goods_pt, label=label_pt, color='C0')

    else:
        label_ct = f"Gini Index: {gini:.3f}"

    plt.plot(cum_population, cum_goods, label=label_ct, color='C1')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Equality Line")
    
    # Add labels and title
    plt.xlabel("Cumulative Share of Population (Ranked by Income)")
    plt.ylabel("Cumulative Share of Goods Bought")
    plt.title("Concentration Curve")
    plt.legend()
    
    # Show plot
    plt.grid(alpha=.3)
    plt.show()


def plot_bought_goods_distribution(
    agents_ct: List[EconomicAgent], 
    agents_pt: List[EconomicAgent]
) -> None:
    """
    Plots the distributions of the average and tota number 
    of bought goods by income bin.

    Args:
        agents_ct (list): economic agents with constant taxation.
        agents_pt (list): economic agents with progressive taxation.

    Returns:
        None
    """
    tot_goods_by_inc = get_cons_ineq_data(agents_ct, agents_pt)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    legend = ["Progressive Taxation", "Constant Taxation"]
    _ = (
        tot_goods_by_inc
        .set_index("income_bin")[["av_goods_pt", "av_goods_ct"]]
        .dropna()
        .plot
        .bar(ax=ax[0], ec="k", zorder=2)
    )
    _ = ax[0].set_xticklabels(
        ax[0].get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    _ = ax[0].legend(legend)
    _ = ax[0].set(
        xlabel="Income Class", 
        ylabel="Avg. # of Goods Bought", 
        title="Avg. # of Goods Bought by Income Class"
    )
    _ = ax[0].grid(alpha=.3)

    _ = (
        tot_goods_by_inc
        .set_index("income_bin")[["goods_pt", "goods_ct"]]
        .plot
        .bar(ax=ax[1], ec="k", zorder=2)
    )
    _ = ax[1].legend(legend)
    _ = ax[1].grid(alpha=.3, zorder=-2)
    _ = ax[1].set_xticklabels(
        ax[1].get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    _ = ax[1].set(
        xlabel="Income Class", 
        ylabel="Tot. # of Goods Bought", 
        title="Tot. # of Goods Bought by Income Class"
    )

    _ = fig.set_tight_layout(True)
