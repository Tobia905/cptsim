from typing import List

import pandas as pd

from cptsim.tax import progressive_tax, redistribute_funds


class EconomicAgent:

    def __init__(
        self, 
        income: int | float,
        min_tax: float = 0.15, 
        max_tax: float = .35, 
        min_inc: int | float = 500, 
        max_inc: int | float = 15000, 
        prog_rate: float = 0.0003,
        saving_rate: float = .20,
        constant_tax: bool = False
    ):
        self.income = income
        self.tax = (
            progressive_tax(
                self.income, min_tax, max_tax, min_inc, max_inc, k=prog_rate
            )
            if not constant_tax else .22
        )
        self.available_income = self.income * (1 - saving_rate)
        self.bought_goods = 0

    def buy_consumption_good(self, price: float):
        self.available_income -= price + (self.tax * price)
        self.bought_goods += 1


class SocialPlanner:

    def __init__(self, agents: List[EconomicAgent], budget: int | float):
        self.agents = agents
        self.budget = budget

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

        return pd.DataFrame(
            {
                "tax": taxes, "income": incomes
            }
        )
    
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

        """
        tax_df = self.get_taxation_df(self.agents)
        tax_df_low = tax_df[tax_df["tax"] <= tax_tresh].copy()

        refound = redistribute_funds(
            tax_df_low["income"], 
            self.budget, 
            decay_rate=progressive_rate
        )
        tax_df_low["income"] += refound
        tax_df.loc[tax_df.index.isin(tax_df_low.index.tolist()), "income"] = tax_df_low["income"]
        return tax_df
