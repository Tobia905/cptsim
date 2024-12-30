from typing import Dict, Union, Tuple, List, Any

import numpy as np

from cptsim.agent import EconomicAgent
from cptsim.consumption_goods import truncated_sampling


def simulate_market(
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]], 
    agents: int = 5000, 
    constant_tax: bool = False,
    **kwargs: Any
) -> Tuple[List[EconomicAgent], float]:
    
    gov_budget = 0
    all_agents = []
    for _ in range(agents):
        ea = EconomicAgent(constant_tax=constant_tax, **kwargs)
        upper_bound = 800 if ea.available_income > 800 else ea.available_income
        good_price = truncated_sampling(
            prices_distribution["distribution"], 
            prices_distribution["parameters"], 
            size=1, 
            lower_bound=1, 
            upper_bound=upper_bound
        )
        while ea.available_income > good_price:
            ea.buy_consumption_good(price=good_price)
            upper_bound = 800 if ea.available_income > 800 else ea.available_income
            good_price = truncated_sampling(
                prices_distribution["distribution"], 
                prices_distribution["parameters"], 
                size=1, 
                lower_bound=1, 
                upper_bound=upper_bound
            )
            gov_budget += ea.tax * good_price

        else:
            all_agents.append(ea)
            continue

    return all_agents, gov_budget
