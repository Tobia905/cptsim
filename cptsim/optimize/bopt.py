from typing import Optional, Dict, Union, Tuple, Any
from functools import partial

import optuna
import torch
import numpy as np
from optuna.study.study import Study
from optuna.integration import BoTorchSampler
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

from cptsim.market import extra_budget_loss
from cptsim.income import simulate_income


class GPWithNoise(ExactGP):
    def __init__(
        self, 
        x_train: torch.Tensor, 
        y_train: torch.Tensor, 
        likelihood: GaussianLikelihood
    ):
        super(GPWithNoise, self).__init__(x_train, y_train, likelihood)
        
        # Use a constant mean to model non-zero noise mean
        self.mean_module = ConstantMean()
        
        # Use an RBF kernel for smoothness
        self.covar_module = RBFKernel()
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return torch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianExtrabudgetOptimizer:

    def __init__(
        self, 
        prices_distribution: Dict[str, Union[np.ndarray, Tuple, float]],
        agents_per_step: int = 1000,
        n_startup_trials: int = 10, 
        random_state: int = None,
        min_tax_range: Tuple[float, float] = (0.15, 0.20),
        max_tax_range: Tuple[float, float] = (0.30, 0.38),
        prog_rate_range: Tuple[float, float] = (0.0001, 0.001),
        **ml_estimator_kwargs: Any
    ):
        
        gp_model = partial(self.create_gp_model, **ml_estimator_kwargs)

        sampler = BoTorchSampler(
            gp_model=gp_model, # Custom GP model
            n_startup_trials=n_startup_trials, # Number of random trials before using GP
            seed=random_state # Random seed for reproducibility
        )
        self.study = optuna.create_study(sampler=sampler)
        self.prices_distribution = prices_distribution
        self.agents_per_step = agents_per_step
        self.min_tax_range = min_tax_range
        self.max_tax_range = max_tax_range
        self.prog_rate_range = prog_rate_range

    def create_gp_model(self, x_train: torch.Tensor, y_train: torch.Tensor):
        likelihood = GaussianLikelihood()

        # Create and fit the GP model
        gp = GPWithNoise(x_train, y_train, likelihood)
        mll = ExactMarginalLogLikelihood(likelihood, gp)
        self.fit_gpytorch_model(mll)
        return gp
    
    @staticmethod
    def fit_gpytorch_model(
        mll: ExactMarginalLogLikelihood, 
        epochs: int = 50, 
        learning_rate: float = 0.1
    ) -> None:
        """
        Fits a GPyTorch model using the Adam optimizer.

        Args:
            mll (ExactMarginalLogLikelihood): Marginal log-likelihood of the GP model.
            epochs (int): the steps for the training loop
            learning_rate (float): 
        """
        # Use the model and likelihood from the MLL
        model = mll.model

        # set the model in training mode
        model.train()
        mll.train()

        # define Adam optimizer
        optimizer = Adam(model.parameters(), lr=learning_rate)

        # Training loop: maximize the marginal likelihood
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

    def objective_function(self, trial: optuna.trial.Trial) -> float:
        # Suggest parameters (minimum tax, maximum tax, growth rate)
        min_tax = trial.suggest_float(
            "min_tax", self.min_tax_range[0], self.min_tax_range[1]
        )
        max_tax = trial.suggest_float(
            "max_tax", self.max_tax_range[0], self.max_tax_range[1]
        )
        prog_rate = trial.suggest_float(
            "prog_rate", self.prog_rate_range[0], self.prog_rate_range[1]
        )
        
        incomes = simulate_income(n=self.agents_per_step)
        # Run the simulation for the given parameters
        extrabudget = extra_budget_loss(
            self.prices_distribution,
            initial_incomes=incomes,
            min_tax=min_tax, 
            max_tax=max_tax, 
            prog_rate=prog_rate
        )
        
        return extrabudget
    
    def run(self, n_trials: int = 100) -> Study:
        _ = self.study.optimize(n_trials=n_trials)
        return self.study
