from typing import List

import pandas as pd
import numpy as np

from cptsim.tax import redistribute_funds
from cptsim.agents.economic_agent import EconomicAgent


class SocialPlanner:

    def __init__(self, agents: List[EconomicAgent], budget: int | float):
        self.agents = agents
        self.budget = budget
        self.budgets = [budget]
        self.total_budget = budget

    def reassign_new_agents_and_budget(self, new_agents: List[EconomicAgent], new_budget: int | float) -> None:
        self.agents = new_agents
        self.budget = new_budget
        self.budgets.append(self.budget)
        self.total_budget += self.budget

    @staticmethod
    def get_taxation_df(agents: List[EconomicAgent]) -> pd.DataFrame:
        """
        Turns a list of agents into a dataframe with
        individual incomes and taxations.

        Args:
            agents (list): a list of economic agents.

        Return:
            pd.DataFrame: a dataframe with incomes and their 
            associated taxation.
        """
        incomes = [ag.income for ag in agents]
        taxes = [ag.tax for ag in agents]

        return pd.DataFrame({"tax": taxes, "income": incomes})
    
    def reassing_extra_budget(
        self,
        tax_tresh: float = .22,
        progressive_rate: float = 0.005
    ) -> pd.DataFrame:
        """
        Reassigns the extra budget gathered by the government
        with respect to the constant taxation scenario.

        Args:
            agents (list): a list of economic agents
            budget (int or float): the total amount that the 
            government can redistribute.
            tax_tresh (float): the threshold above which the 
            redistribution stops.
            progressive_rate (float): the decay rate of the 
            exponential redistribution function.

        Returns:
            tax_df (pd.DataFrame): a dataframe with individual 
            taxations and incomes after the redistribution.
        """
        tax_df = self.get_taxation_df(self.agents)

        # proceed with the redistribution only if the budget is
        # above 0
        if self.budget > 0:
            tax_df_low = tax_df[tax_df["tax"] <= tax_tresh].copy()

            refound = redistribute_funds(
                tax_df_low["income"], 
                self.budget, 
                decay_rate=progressive_rate
            )
            # assing the refound ad obtain the original income dataframe
            tax_df_low["income"] += refound
            tax_df.loc[
                tax_df.index.isin(tax_df_low.index.tolist()), "income"
            ] = tax_df_low["income"]

        return tax_df
    
    def get_agent_attribute(self, attribute: str) -> List[int | float | List[int | float]]:
        return [eval(f"ag.{attribute}") for ag in self.agents]

    def get_economy_df(self) -> pd.DataFrame:
        """
        Represents the list of agents as a dataframe.

        Returns:
            ag_df (pd.DataFrame): the agent dataframe representing
            the whole economy.
        """
        ag_df = pd.DataFrame(
            {
                "agent": np.arange(len(self.agents)),
                "initial_income": self.get_agent_attribute("income"),
                "final_income": self.get_agent_attribute("available_income"),
                "tax": self.get_agent_attribute("tax"),
                "paid_taxes": self.get_agent_attribute("paid_taxes"),
                "bought_goods": self.get_agent_attribute("bought_goods"),
                "paid_prices": self.get_agent_attribute("paid_prices"),
                "savings": self.get_agent_attribute("savings"),
                "income_dynamic": self.get_agent_attribute("income_dynamic")
            }
        )
        return ag_df