# cptsim #

## Overview ##

A simulation study on the introduction of a continuous function based progressive tax on consumption goods.

Our hypothesis is that a taxation designed like this would reduce income inequality while having a positive effect on the total budget gathered by the government.

To test the hypotesis, we developed an agent based simulation in which we simulate a market with two kind of agents: the Economic Agent, represented by the class **EconomicAgent** and the Social Planner, represented by the class **SocialPlanner**. The code for both the two economic entities can be found inside **cptsim/agent.py**. 

## One Step Simulation ##

### 1. The Model ###

The idea is straightforward: at time $t$, we generate $n$ agents and we assign them an income ($Y$) sampled from an heavy tailed Gamma distribution. Then, we let them buy $m$ consumption goods untill they reach $(1 - \gamma) \cdot Y$, where $\gamma$ is the saving rate, that we assume constant across individuals and equal to 0.20 (20%). We do this both for a "constant taxation" scenario - with a consumption tax always equal to 0.22 (22%) and for a "progressive taxation" one, where the taxation is a continuous "rescaled" sigmoid function of the individual income, defined as follows: 

$$\omega_i = \omega_{min} + \frac{\omega_{\text{max}} - \omega_{\text{min}}}{1 + e^{-k \cdot (Y - Y_{\text{mid}})}}$$

Where:
- $\omega$: Tax rate
- $Y$: Income
- $\omega_{min}$: Minimum tax rate
- $\omega_{max}$: Maximum tax rate
- $Y_{min}$: Minimum income
- $Y_{max}$: Maximum income
- $k$: Progressive factor
- $Y_{mid} = \frac{Y_{min} + Y_{max}}{2}$: Midpoint income

A scaled sigmoid function has some interesting properties that make it a good candidate for simulating a progressive tax:
- First, the scaling procedure allows to consider a lower and an upper limit.
- It naturally maps data from $R$ to $[0, 1]$.
- Maximum and minimum values can be selected.
- The progressive factor allows to consider different levels of steepness and, hence, progressivity.

Then, we compare the two scenarios in terms of budget gathered by the Social Planner: if the progressive taxation is associated with an extra budget w.r.t. the constant one, at $t + 1$ this is redistributed to lower income individuals following a "left tail adjustment" strategy: we use exponential decay to assign the proportion of the total budget as a continuos function of the income: 

$$R_i = \frac{e^{-\lambda \cdot Y_i}}{\sum_{j} e^{-\lambda \cdot Y_j}} \cdot F$$

Where:
- $R_i$: Redistributed funds for individual $i$
- $Y_i$: Income of individual $i$
- $F$: Total funds available for redistribution
- $\lambda$: Decay rate (higher values prioritize lower incomes)
- $e^{-\lambda \cdot Y_i}$: Weight assigned to individual $i$ based on their income
- $\sum_{j} e^{-\lambda \cdot Y_j}$: Sum of weights for normalization

The code for the taxation and redistribution functions can be found in **cptsim/tax.py**, while the code for the simulation can be found in **cptsim/simulation.py**.

### 2. Redistribution Evaluation ###

At the end of the simulation we evaluate the redistribution under two different dimensions:

 - **Direct Income Redistribution**: we compare the pre and post redistribution income distributions and we compute the respective Lorenz curves and the gini indexes as follows: 
 
    $$G = 1 - 2 \int_0^1 L(x)dx \approx \frac{2 \sum_{i=1}^n i x_i}{n \sum_{i=1}^n x_i} - \frac{n + 1}{n}$$

    Where:
    - $L(x)$ is the Lorenz curve, which represents the cumulative share of income (or wealth) as a function of the cumulative share of the population.
    - $x_i$ represents the income of the $i$-th individual, sorted in ascending order.
    - $n$ represents the total number of individuals.

 - **Consumption Inequality**: here we assume that a reduction in consumption inequality, if present, can be considered as an emerging property of the system. In fact, the progressive differences in taxation for different level of incomes will reduce the amount paid for the same number of consumption goods bought by the lower income individuals and, hence, increase the amount of goods bought. To measure it we consider two dimensions:
    - We use a variation of the Lorenz curve in which we relate the cumulative share of goods bought with the share of population ranked by income.
    - We do the same with the amount of tax paid for the same number of goods bought.

The code for the evaluation functions can be found in **cptsim/reporting/income_inequality.py** and **cptsim/reporting/consumption_inequality.py**.

## Multi Step Simulation ##

This part is yet to be implemented. The idea is generating an initial population of agents with a certain income and a certain tax rate and repeating the one step simulation a certain number of times. For each step, we can assume that the lower income individuals spend or save their redistributed share, also considering a combination of the two.
At the end of the simulation, we can compare the metrics listed above for the overall process, considering, for example, the savings or the consumption inequality.