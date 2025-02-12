from typing import Dict, Union, Tuple, Any, List
from typing_extensions import Self

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import NotFittedError
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real

from cptsim.market import extra_budget_loss
from cptsim.income import simulate_income
from cptsim.tax import progressive_tax


class BayesianExtrabudgetOptimizer:

    def __init__(
        self: Self,
        prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]],
        agents_per_step: int = 1000,
        min_income=500,
        max_income=15000,
        heavy_tail_factor=1.5,
        random_state: int = None,
        min_tax_range: Tuple[float, float] = (0.15, 0.20),
        max_tax_range: Tuple[float, float] = (0.30, 0.38),
        prog_rate_range: Tuple[float, float] = (0.0001, 0.001),
        n_jobs: int = -1
    ) -> None:

        self.prices_distribution = prices_distribution
        self.agents_per_step = agents_per_step
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.min_income = min_income
        self.max_income = max_income
        self.heavy_tail_factor = heavy_tail_factor

        self.search_space = [
            # Range for min tax
            Real(min_tax_range[0], min_tax_range[1], name="min_tax"),
            # Range for max tax
            Real(max_tax_range[0], max_tax_range[1], name="max_tax"),
            # Range for the progressive rate
            Real(prog_rate_range[0], prog_rate_range[1], name="prog_rate")
        ]

    def run(
        self: Self,
        n_startup_trials: int = 10,
        iterations: int = 50,
        noise: str = "gaussian",
        aq_function: str = "gp_hedge",
        **optimizer_kwargs: Any
    ) -> Self:

        incomes = simulate_income(
            n=self.agents_per_step,
            min_income=self.min_income,
            max_income=self.max_income,
            heavy_tail_factor=self.heavy_tail_factor
        )

        @use_named_args(self.search_space)
        def objective(
            min_tax: float,
            max_tax: float,
            prog_rate: float,
            **kwargs
        ) -> float:

            extrabudget = extra_budget_loss(
                self.prices_distribution,
                initial_incomes=incomes,
                min_tax=min_tax,
                max_tax=max_tax,
                prog_rate=prog_rate,
                **kwargs
            )
            return -extrabudget[0]

        self.result_ = gp_minimize(
            func=objective,
            dimensions=self.search_space,
            n_calls=iterations, # Number of function evaluations
            n_random_starts=n_startup_trials, # Number of random initial points
            acq_func=aq_function, # Acquisition function to balance exploitation/exploration
            noise=noise, # Noise level in the optimization
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **optimizer_kwargs
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.__check_is_fitted()
        return self.result_.specs

    def plot_optimal_taxation(self: Self) -> None:
        self.__check_is_fitted()
        incomes = simulate_income(
            n=self.agents_per_step,
            min_income=self.min_income,
            max_income=self.max_income,
            heavy_tail_factor=self.heavy_tail_factor
        )

        min_t, max_t, prog_r = self.best_params
        cons_tax = progressive_tax(
            incomes, min_t, max_t, self.min_income, self.max_income, k=prog_r
        )
        plt.axhline(.22, color="r", zorder=2)
        plt.plot(sorted(incomes), sorted(cons_tax), c="C0", zorder=1)
        plt.legend(["Constant Tax", "Optimal Progressive Tax"])
        plt.title("Optimal Progressive Consumption Tax")
        plt.xlabel("Post-Taxation Monthly Income")
        plt.ylabel("Tax")
        plt.grid(alpha=.3, zorder=-2)
        plt.show()

    def __check_is_fitted(self: Self) -> None:
        if not hasattr(self, "result_"):
            raise NotFittedError

    @property
    def best_params(self: Self) -> List[float]:
        self.__check_is_fitted()
        return self.result_.x

    @property
    def trials(self: Self) -> List[List[float]]:
        self.__check_is_fitted()
        return self.result_.x_iters

    @property
    def min_obj(self: Self) -> float:
        self.__check_is_fitted()
        return self.result_.fun
