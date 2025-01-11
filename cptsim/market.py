from typing import Dict, Union, Tuple, List, Any, Optional, Literal
import logging
from copy import deepcopy
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from cptsim.agents.economic_agent import EconomicAgent
from cptsim.agents.social_planner import SocialPlanner
from cptsim.consumption_goods import truncated_sampling
from cptsim.income import simulate_income
from cptsim.utils import match_kwargs, check_threads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_upper_bound(
    upper_bound: float, 
    income: int | float, 
    tax: float
) -> float:
    # set the income as upper bound if the first is
    # lower than the second
    income = deepcopy(income)
    upper_bound_ = (
        upper_bound 
        if income > upper_bound
        else income
    )
    # subtract the taxation if needed
    if income < upper_bound * (1 + tax):
        upper_bound_ -= (upper_bound_ * tax)

    return upper_bound_


def _compute_refound_from_income_df(
    income_df: pd.DataFrame, original_incomes: ArrayLike
) -> List[float]:
    refounds = income_df["income"].values - np.array(original_incomes)
    refounds = refounds if np.sum(refounds) > 0 else None
    return refounds


def simulate_agent(
    n: int, 
    inc: float, 
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]],
    initial_agents: List[EconomicAgent], 
    gov_transfer: float, 
    constant_tax: bool = False, 
    lower_bound: int = 1, 
    upper_bound: int = 1000, 
    verbose: bool = False, 
    **kwargs: Any
) -> Tuple[EconomicAgent, float]:
    
    gov_budget = 0
    # initialize the agent
    if not initial_agents:
        ea = EconomicAgent(
            income=inc, 
            constant_tax=constant_tax, 
            **match_kwargs(EconomicAgent, kwargs)
        )

    # use the provided list of agents
    else:
        ea = initial_agents[n]
        transfer = gov_transfer[n] if gov_transfer is not None else gov_transfer
        ea.reassing_new_income(inc, transfer=transfer)

    # compute initial upper bound and sample the good price
    upper_bound_ = _compute_upper_bound(upper_bound, ea.available_income, ea.tax)
    good_price = truncated_sampling(
        prices_distribution["distribution"], 
        prices_distribution["parameters"], 
        size=1, 
        lower_bound=lower_bound, 
        upper_bound=upper_bound_
    )

    # keep buying consumption goods untill possible
    while ea.available_income >= good_price:
        ea.buy_consumption_good(
            price=good_price, **match_kwargs(ea.buy_consumption_good, kwargs)
        )
        gov_budget += ea.tax * good_price
        # dynamically adjust the upper bound
        upper_bound_ = _compute_upper_bound(upper_bound, ea.available_income, ea.tax)

        # sample new good's price
        if upper_bound_ > lower_bound:
            good_price = truncated_sampling(
                prices_distribution["distribution"], 
                prices_distribution["parameters"], 
                size=1, 
                lower_bound=lower_bound, 
                upper_bound=upper_bound_
            )

        # the available income is over, store the agent and break 
        # the while loop
        else:
            if verbose:
                logger.info(
                    f"Agent {n}"
                    f" | initial income: {ea.income}"
                    f" | saved income: {ea.saving_rate * ea.income}"
                    f" | initial available income: {ea.income * (1 - ea.saving_rate)}"
                    f" | goods bought: {ea.bought_goods}"
                    f" | total spending: {sum(ea.paid_prices)}"
                    f" | total taxes paid: {sum(ea.paid_taxes)}"
                    f" | final income: {ea.available_income}"
                )
            break

    return ea, gov_budget


def simulate_market(
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]], 
    initial_incomes: Optional[ArrayLike] = None,
    initial_agents: Optional[List[EconomicAgent]] = None,
    agents: int = 5000, 
    constant_tax: bool = False,
    lower_bound: int = 1,
    upper_bound: int = 1000,
    verbose: bool = False,
    gov_transfer: Optional[ArrayLike] = None,
    n_jobs: int = 1,
    **kwargs: Any
) -> Tuple[List[EconomicAgent], float]:
    """
    Simulates one step (t) of the market. Each economic agent
    saves the x% of his budget and then buys consumption goods 
    utill he finishes his available income. For each good, the 
    government collects taxes and increases his budget.

     Args:
        prices_distribution (dict): dictionary representing the 
        prices distribution.
        initial_agents (ArrayLike): optional array-like of agents.
        initial_incomes (ArrayLike): optional array-like of incomes.
        agents (int): the number of agents to generate (used only if
        initial_agents is None).
        constant_tax (bool): if True, a constant taxation is used.
        lower_bound (int): the lower bound for the truncation in the
        sampling of the consumption goods.
        upper_bound (int): the upper bound for the truncation in the
        sampling of the consumption goods.
        verbose (bool): wheter to show information about the simulation
        or not.
        gov_transfer (float): the amount to be redistributed.
        n_jobs (int): the number of jobs to use in parallelization.

    Returns:
        tuple: the list of agents after the process and the government
        budget. 
    """
    # logs won't be displayed so we set verbose to False
    if n_jobs > 1 or n_jobs == -1:
        verbose = False

    # simulate all incomes if not provided as argument
    incomes = (
        simulate_income(
            n=agents, **match_kwargs(simulate_income, kwargs)
        )
        if initial_incomes is None
        else initial_incomes
    )

    # check the provided number of threads
    n_jobs = check_threads(num_threads=n_jobs)
    # iterate over the incomes in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_agent)(
            n, 
            inc, 
            prices_distribution,
            initial_agents=initial_agents, 
            gov_transfer=gov_transfer, 
            constant_tax=constant_tax,  
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            verbose=verbose, 
            **kwargs
        )
        for n, inc in enumerate(incomes)
    )

    # unzip results
    all_agents, gov_budgets = zip(*results)
    # sum all budgets gathered by the government
    total_gov_budget = sum(gov_budgets)

    return list(all_agents), total_gov_budget


def simulate_market_repeated(
    steps: int,
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]], 
    initial_incomes: Optional[ArrayLike] = None,
    agents: int = 5000, 
    lower_bound: int = 1,
    upper_bound: int = 1000,
    verbose: bool = False,
    redistribution_handling_strategy: Literal["save", "spend", "mixed"] = "save",
    **kwargs: Any
) -> Tuple[Dict[str, Dict[str, ArrayLike | float]], SocialPlanner]:
    """
    Multistep simulation with agents interaction: simulate_market
    is performed a number of times uqual to steps to simulate
    more than one period of time and obtaing dynamical aggregation
    of the economic variables of the system (e.g. income or the
    government budget). At each step, the market is simulate and
    the SocialPlanner budget is gathered. Then, the SocialPlanner
    redistributes the extrabudget, gathered w.r.t. the constant
    taxation scenario, according to the exponential decay law.
    Here, we assume that the extrabudget is redistributed to all
    individuals which pay less than 22% of taxation, but this can
    be easily parametrized in the function keyword args.

    Args:
        steps (int): the number of times to repeat the simulation.
        prices_distribution (dict): dictionary representing the 
        prices distribution.
        initial_agents (ArrayLike): optional array-like of agents.
        initial_incomes (ArrayLike): optional array-like of incomes.
        agents (int): the number of agents to generate (used only if
        initial_agents is None).
        lower_bound (int): the lower bound for the truncation in the
        sampling of the consumption goods.
        upper_bound (int): the upper bound for the truncation in the
        sampling of the consumption goods.
        verbose (bool): wheter to show information about the simulation
        or not.
        redistribution_handling_strategy (Literal): how the agents handle
        the redistributed income.

    Returns:
        final_results, sp (tuple): a dictionary storing the results for 
        each step and the SocialPlanner class. 
    """
    if redistribution_handling_strategy not in ["save", "spend", "mixed"]:
        raise ValueError(
            "redistribution_handling_strategy must be one of ('save', 'spend', 'mixed')"
        )
    
    if redistribution_handling_strategy == "mixed":
        raise NotImplementedError
    
    # we need to generate income outside the one step simulation
    # to insure coherence with incomes and agents
    if initial_incomes is None:
        initial_incomes = (
            simulate_income(
                n=agents, **match_kwargs(simulate_income, kwargs)
            )
            if initial_incomes is None
            else initial_incomes
        )

    # simulating the first step: the market process takes place 
    # and the social planner redistributes the extra-budget 
    # (if available)
    initial_agents_pt, initial_gov_budget_pt = simulate_market(
        prices_distribution,
        initial_incomes=initial_incomes,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        verbose=verbose,
        transfer_handling_strategy=redistribution_handling_strategy,
        **kwargs
    )
    # repeat the step with constant taxation
    initial_agents_ct, initial_gov_budget_ct = simulate_market(
        prices_distribution,
        initial_incomes=initial_incomes,
        constant_tax=True,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        verbose=verbose,
        transfer_handling_strategy=redistribution_handling_strategy,
        **kwargs
    )
    # compute the extra-budget from the progressive taxation
    extra_budget = initial_gov_budget_pt - initial_gov_budget_ct

    sp = SocialPlanner(
        agents=initial_agents_pt, budget=extra_budget
    )
    # we adjust the initial income with the redistribution and
    # we compute the actual refound
    refounds = _compute_refound_from_income_df(
        sp.reassing_extra_budget(
            **match_kwargs(sp.reassing_extra_budget, kwargs)
        ),
        initial_incomes
    )
    final_results = {"agents_pt": {}, "agents_ct": {}, "gov_budgets": {}}
    # then we start the multi-step simulation loop
    for step in steps:
        # progressive taxation
        curr_agents_pt, curr_gov_budget_pt = simulate_market(
            prices_distribution,
            initial_agents=initial_agents_pt,
            initial_incomes=initial_incomes,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            verbose=verbose,
            gov_transfer=refounds,
            **kwargs
        )
        # constant taxation
        curr_agents_ct, curr_gov_budget_ct = simulate_market(
            prices_distribution,
            initial_agents=initial_agents_ct,
            initial_incomes=initial_incomes,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            verbose=verbose,
            gov_transfer=None,
            **kwargs
        )
        # compute the current extra-budget
        extra_budget = curr_gov_budget_pt - curr_gov_budget_ct
        initial_agents_pt = curr_agents_pt
        initial_agents_ct = curr_agents_ct
        # update the social planner
        _ = sp.reassign_new_agents_and_budget(initial_agents_pt, extra_budget)
        refounds = _compute_refound_from_income_df(
            sp.reassing_extra_budget(
                **match_kwargs(sp.reassing_extra_budget, kwargs)
            ),
            initial_incomes
        )

        # store the results for each step
        final_results["agents_pt"][step] = curr_agents_pt
        final_results["agents_ct"][step] = curr_agents_ct
        final_results["gov_budgets"][step] = extra_budget

    return final_results, sp


def extra_budget_loss(
    prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]],
    agents: int = 300, 
    min_tax: float = .15, 
    max_tax: float = .38, 
    heavy_tail_factor: float = 1.5, 
    prog_rate: float = 0.00015, 
    implicit_tax: bool = False, 
    verbose: bool = False,
    n_jobs: int = 5,
    **kwargs: Any,
) -> float:
    
    initial_incomes = simulate_income(
        n=agents, **match_kwargs(simulate_income, kwargs)
    )
    _, gov_budget_pt = simulate_market(
        prices_distribution,
        initial_incomes=initial_incomes,
        implicit_tax=implicit_tax,
        n_jobs=n_jobs,
        verbose=verbose,
        min_tax=min_tax,
        max_tax=max_tax,
        prog_rate=prog_rate,
        heavy_tail_factor=heavy_tail_factor,
        **kwargs
    )
    _, gov_budget_ct = simulate_market(
        prices_distribution,
        initial_incomes=initial_incomes,
        implicit_tax=implicit_tax,
        n_jobs=n_jobs,
        verbose=verbose,
        constant_tax=True,
        min_tax=min_tax,
        max_tax=max_tax,
        prog_rate=prog_rate,
        heavy_tail_factor=heavy_tail_factor,
        **kwargs
    )
    return - (gov_budget_pt - gov_budget_ct)
