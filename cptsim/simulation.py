from typing import Dict, Union, Tuple, List, Any, Optional
import logging

import numpy as np
from numpy.typing import ArrayLike

from cptsim.agent import EconomicAgent
from cptsim.consumption_goods import truncated_sampling
from cptsim.income import simulate_income
from cptsim.utils import match_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_upper_bound(
    upper_bound: float, 
    income: int | float, 
    tax: float
) -> float:
    upper_bound_ = (
        upper_bound 
        if income > upper_bound
        else income
    )
    if income < upper_bound * (1 + tax):
        upper_bound_ -= (upper_bound_ * tax)

    return upper_bound_


def simulate_market(
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]], 
    initial_incomes: Optional[ArrayLike] = None,
    initial_agents: Optional[List[EconomicAgent]] = None,
    agents: int = 5000, 
    constant_tax: bool = False,
    lower_bound: int = 1,
    upper_bound: int = 1000,
    verbose: bool = False,
    **kwargs: Any
) -> Tuple[List[EconomicAgent], float]:
    
    gov_budget = 0
    all_agents = []
    incomes = (
        simulate_income(
            n=agents, **match_kwargs(simulate_income, kwargs)
        )
        if initial_incomes is None
        else initial_incomes
    )
    for n, inc in enumerate(incomes):
        if not initial_agents:
            ea = EconomicAgent(
                income=inc, 
                constant_tax=constant_tax, 
                **match_kwargs(EconomicAgent, kwargs)
            )

        else:
            ea = initial_agents[n]
            ea.reassing_new_income(inc)

        upper_bound_ = _compute_upper_bound(
            upper_bound, ea.available_income, ea.tax
        )
        good_price = truncated_sampling(
            prices_distribution["distribution"], 
            prices_distribution["parameters"], 
            size=1, 
            lower_bound=lower_bound, 
            upper_bound=upper_bound_
        )
        while ea.available_income >= good_price:
            ea.buy_consumption_good(price=good_price)
            gov_budget += ea.tax * good_price
            upper_bound_ = _compute_upper_bound(
                upper_bound, ea.available_income, ea.tax
            )
            if upper_bound_ > lower_bound:
                good_price = truncated_sampling(
                    prices_distribution["distribution"], 
                    prices_distribution["parameters"], 
                    size=1, 
                    lower_bound=lower_bound, 
                    upper_bound=upper_bound_
                )

            else:
                if verbose:
                    logger.info(
                        f"Agent {n}"
                        f" | initial income: {ea.income}"
                        f" | goods bought: {ea.bought_goods}"
                        f" | total spending: {sum(ea.paid_prices)}"
                        f" | total taxes paid: {sum(ea.paid_taxes)}"
                        f" | final income: {ea.available_income}"
                    )
                all_agents.append(ea)
                break

    return all_agents, gov_budget
