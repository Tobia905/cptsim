from cptsim.income import simulate_income
from cptsim.tax import progressive_tax


class EconomicAgent:

    def __init__(
        self, 
        min_tax: float = 0.15, 
        max_tax: float = .35, 
        min_inc: int | float = 500, 
        max_inc: int | float = 7500, 
        prog_rate: float = 0.0007,
        saving_rate: float = .20,
        constant_tax: bool = False
    ):
        self.income = simulate_income(n=1)[0]
        self.tax = (
            progressive_tax(
                self.income, min_tax, max_tax, min_inc, max_inc, k=prog_rate
            )
            if not constant_tax else .22
        )
        self.available_income = self.income * (1 - saving_rate)

    def buy_consumption_good(self, price: float):
        self.available_income -= price + (self.tax * price)
