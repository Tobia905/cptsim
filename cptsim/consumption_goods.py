from typing import Dict, Union, Tuple, Any, Optional

import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import entropy


def kl_divergence(
    data_p: ArrayLike, 
    data_q: ArrayLike, 
    bins: int = 50, 
    range: Optional[ArrayLike] = None
) -> float:
    """
    Computes the Kullback-Leibler (KL) divergence between two empirical distributions.

    Args:
        data_p (ArrayLike): Samples from the first distribution (P).
        data_q (ArrayLike): Samples from the second distribution (Q).
        bins (int): Number of bins for histogram-based density estimation.
        range (tuple): The range (min, max) for the histograms. If None, inferred from the data.

    Returns:
        float: The KL divergence.
    """
    # Compute histograms (empirical PDFs)
    hist_p, bin_edges = np.histogram(data_p, bins=bins, range=range, density=True)
    hist_q, _ = np.histogram(data_q, bins=bin_edges, density=True)

    # Avoid division by zero or log(0) by adding a small constant
    hist_p = np.clip(hist_p, 1e-10, None)
    hist_q = np.clip(hist_q, 1e-10, None)

    # Normalize the histograms to ensure they sum to 1
    hist_p /= np.sum(hist_p)
    hist_q /= np.sum(hist_q)

    # Compute the KL divergence
    kl_div = entropy(hist_p, hist_q)

    return kl_div


def truncated_sampling(dist, params, size, lower_bound, upper_bound) -> np.ndarray:
    lower_cdf = dist.cdf(lower_bound, *params)
    upper_cdf = dist.cdf(upper_bound, *params)

    # Ensure CDF bounds are valid
    if lower_cdf >= upper_cdf or not (0 <= lower_cdf <= 1) or not (0 <= upper_cdf <= 1):
        raise ValueError(f"Invalid CDF bounds: lower_cdf={lower_cdf}, upper_cdf={upper_cdf}")
    
    truncated_samples = dist.ppf(
        np.random.uniform(lower_cdf, upper_cdf, size), *params
    )
    return truncated_samples


def fit_and_compare_distributions(
    data: ArrayLike, 
    lower_bound: int | float = 1, 
    upper_bound: int | float = 1000
) -> Dict[str, Union[np.ndarray, Tuple, float]]:
    """
    Fits Log-Normal, Gamma, Weibull, and Exponential distributions to the given data,
    generates samples (restricted between lower_bound and upper_bound), and compares 
    them using the Kolmogorov-Smirnov test.
    
    Args:
        data (ArrayLike): The input data.
        lower_bound (float): Lower bound for the sampling.
        upper_bound (float): Upper bound for the sampling.

    Returns:
        results (dict): the result of the whole fitting process.
    """
    distributions = {
        'Log-Normal': stats.lognorm,
        'Gamma': stats.gamma,
        'Weibull': stats.weibull_min,
        'Exponential': stats.expon
    }

    results = {}

    for name, dist in distributions.items():
        # Fit the distribution to the data
        params = dist.fit(data)

        # Generate samples restricted between [lower_bound, upper_bound]
        samples = truncated_sampling(
            dist, 
            params, 
            size=len(data), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound
        )

        # Compute KL divergence between data and generated samples
        kl_div = kl_divergence(data, samples)

        # Store results
        results[name] = {
            'distribution': distributions[name],
            'samples': samples,
            'parameters': params,
            'kl_divergence': kl_div,
        }

    return results


def get_best_fitting_distribution(results: Dict[str, Any]) -> Dict[str, Any]:
    res_df = pd.DataFrame(results).T
    best_dist = res_df[res_df["kl_divergence"] == res_df["kl_divergence"].min()].index[0]

    return results[best_dist]


def plot_prices_distribution(
    prices: ArrayLike, 
    title: str = "Consumption Goods' Prices Distribution"
):
    prices = pd.Series(prices)
    prices.hist(bins=50, edgecolor="k", grid=False, zorder=2)
    plt.axvline(prices.mean(), c="r", label="Mean")
    plt.axvline(prices.median(), c="g", label="Median")
    plt.xlabel("Price ($)")
    plt.ylabel("#")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=.3, zorder=-2)
