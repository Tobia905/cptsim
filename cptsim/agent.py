from typing import List, Literal, Optional

import pandas as pd
import numpy as np

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
        constant_tax: bool = False,
        transfer_handling_strategy: Literal["save", "spend"] = "save"
    ):
        if transfer_handling_strategy not in ["save", "spend"]:
            raise ValueError(
                "transfer_handling_strategy must be one of ('save', 'spend')"
            )
        self.income = income
        self.saving_rate = saving_rate
        self.tax = (
            progressive_tax(
                self.income, min_tax, max_tax, min_inc, max_inc, k=prog_rate
            )
            if not constant_tax else .22
        )
        self.available_income = self.income * (1 - self.saving_rate)
        self.savings = self.income * self.saving_rate
        self.bought_goods = 0
        self.paid_prices = []
        self.paid_taxes = []
        self.transfers = []
        self.transfer_handling_strategy = transfer_handling_strategy

    def buy_consumption_good(self, price: float) -> None:
        self.available_income -= price + (self.tax * price)
        self.bought_goods += 1
        self.paid_prices.append(price)
        self.paid_taxes.append(self.tax * price)

    def reassing_new_income(self, new_income: int | float, transfer: Optional[int | float] = None) -> None:
        self.income = new_income
        self.handle_redistribution(transfer=transfer)
        self.available_income = self.income * (1 - self.saving_rate)
        self.savings += (self.income * self.saving_rate)

    def handle_redistribution(self, transfer: Optional[int | float] = None) -> None:
        if transfer:
            self.transfers.append(transfer)
            if (
                self.income + transfer > self.income 
                and 
                self.transfer_handling_strategy == "save"
            ):
                self.savings += transfer

            else:
                self.available_income += transfer

    @property
    def income_dynamic(self):
        inc_dyn = []

        if len(self.paid_prices) > 0:
            starting_income = self.income - (self.saving_rate * self.income)
            for price, tax in zip(self.paid_prices, self.paid_taxes):
                starting_income -= (price + tax)
                inc_dyn.append(starting_income)

        return inc_dyn


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

        if self.budget > 0:
            tax_df_low = tax_df[tax_df["tax"] <= tax_tresh].copy()

            refound = redistribute_funds(
                tax_df_low["income"], 
                self.budget, 
                decay_rate=progressive_rate
            )
            tax_df_low["income"] += refound
            tax_df.loc[
                tax_df.index.isin(tax_df_low.index.tolist()), "income"
            ] = tax_df_low["income"]

        return tax_df
    
    def get_agent_attribute(self, attribute: str) -> List[int | float | List[int | float]]:
        return [eval(f"ag.{attribute}") for ag in self.agents]

    def get_economy_df(self) -> pd.DataFrame:

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
