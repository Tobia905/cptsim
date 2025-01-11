from typing import Literal, Optional
from copy import deepcopy

from cptsim.tax import progressive_tax


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
        self.implicit_tax = None

    def buy_consumption_good(self, price: float, implicit_tax: bool = True) -> None:
        # setting implicit_tax to True means that we consider it by
        # adjusting the sampling upper bound for the price distribution
        # by considering the taxation
        self.implicit_tax = implicit_tax
        tax = self.handle_implicit_tax(self.implicit_tax)
        self.available_income -= price + (tax * price)
        self.bought_goods += 1
        self.paid_prices.append(price)
        self.paid_taxes.append(self.tax * price)

        # here we accept the possibility of over-spending (happens only if
        # we set implicit_tax to False)
        if self.available_income < 0:
            self.savings += self.available_income

    def reassing_new_income(
        self, 
        new_income: int | float, 
        transfer: Optional[int | float] = None
    ) -> None:
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

            self.transfers.append(transfer)

    @property
    def income_dynamic(self):

        if len(self.paid_prices) > 0:
            starting_income = self.income - (self.saving_rate * self.income)
            inc_dyn = [starting_income]
            for price in self.paid_prices:
                tax = self.handle_implicit_tax(self.implicit_tax)
                starting_income -= price + (tax * price)
                inc_dyn.append(deepcopy(starting_income)[0])

        return inc_dyn
    
    def handle_implicit_tax(self, implicit_tax: bool = True) -> float:
        return self.tax if not implicit_tax else 0
