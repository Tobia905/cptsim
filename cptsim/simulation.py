from typing import Dict, Union, Tuple, List, Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from cptsim.agent import EconomicAgent
from cptsim.consumption_goods import truncated_sampling
from cptsim.income import simulate_income
from cptsim.utils import match_kwargs


def simulate_market(
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]], 
    initial_incomes: Optional[ArrayLike] = None,
    initial_agents: Optional[List[EconomicAgent]] = None,
    agents: int = 5000, 
    constant_tax: bool = False,
    lower_bound: int = 1,
    upper_bound: int = 1000,
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
            
        upper_bound_ = (
            upper_bound 
            if ea.available_income > upper_bound 
            else ea.available_income
        )
        good_price = truncated_sampling(
            prices_distribution["distribution"], 
            prices_distribution["parameters"], 
            size=1, 
            lower_bound=lower_bound, 
            upper_bound=upper_bound_
        )
        while ea.available_income > good_price and ea.available_income > 1:
            ea.buy_consumption_good(price=good_price)
            upper_bound_ = (
                upper_bound_ 
                if ea.available_income > upper_bound_ 
                else ea.available_income
            )
            good_price = truncated_sampling(
                prices_distribution["distribution"], 
                prices_distribution["parameters"], 
                size=1, 
                lower_bound=lower_bound, 
                upper_bound=upper_bound
            )
            gov_budget += ea.tax * good_price

        else:
            all_agents.append(ea)
            continue

    return all_agents, gov_budget
