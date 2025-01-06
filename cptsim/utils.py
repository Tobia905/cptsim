from typing import Callable, Dict, Any
import inspect

import pandas as pd
import numpy as np


def get_function_kwargs(func: Callable[..., Any]) -> list[str]:
    # get function parameters
    pars_dict = dict(inspect.signature(func).parameters)
    # extract only keyword arguments
    return [key for key, val in pars_dict.items() if "=" in val.__str__()]


def match_kwargs(func: Callable[..., Any], kwargs) -> Dict[str, Any]:
    func_kwargs = get_function_kwargs(func)
    passed = list(kwargs.keys())
    # match function kwargs with passsed ones
    return {key: kwargs[key] for key in passed if key in func_kwargs}


def prettify_pandas_bins(binned_values: pd.Series) -> pd.Series:
    return (
        binned_values
        .astype(str)
        .str.replace(r"(\(|\])", "", regex=True)
        .str.replace(", ", " - ")
    )


def CDF(data):

    # data must be a series
    data = pd.Series(data)
    data = data.dropna().reset_index(drop=True)

    # sort the data
    data_sorted = data.sort_values().reset_index(drop=True)

    # calculate the proportional values of samples
    return 1. * np.arange(len(data)) / (len(data) - 1), data_sorted
