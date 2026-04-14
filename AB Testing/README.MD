# A/B Testing using Multi-Armed Bandits

## Problem Description
This project simulates an A/B testing scenario using a Multi-Armed Bandit framework.
Each arm represents an advertisement option with unknown reward distribution. The goal is to maximize cumulative reward while minimizing regret.

---

## Bandits (Ads)
We simulate 4 advertisement options with true mean rewards:

- Arm 1: 1
- Arm 2: 2
- Arm 3: 3
- Arm 4: 4

Rewards are sampled from a Gaussian distribution:
N(mean, 1)

---

## Algorithms

### 1. Epsilon-Greedy
- Exploration probability: ε = 1/t (decaying over time)
- Exploits best-known arm otherwise

### 2. Thompson Sampling
- Bayesian approach using Gaussian likelihood
- Known precision (fixed variance)
- Samples from posterior to select actions

---

## Experiment Setup
- Number of arms: 4
- Number of trials: 20,000
- Reward distribution: Gaussian (μ = [1,2,3,4], σ = 1)
- Noisy stochastic environment

---

## Evaluation Metrics

### Cumulative Reward
Sum of all obtained rewards over time.

### Regret
Difference between optimal expected reward and chosen arm reward:
Regret = μ* − μ_selected

---

## Output Files
- epsilon_greedy.csv
- thompson_sampling.csv
- combined_results.csv

Each file contains:
- Bandit
- Reward
- Algorithm

---

## Results Summary
Thompson Sampling converges faster and achieves lower cumulative regret compared to Epsilon-Greedy due to better exploration-exploitation balance.