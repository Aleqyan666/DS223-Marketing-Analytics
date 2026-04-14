import numpy as np
import pandas as pd
from bandit_base import Bandit
from loguru import logger
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class ThompsonSampling(Bandit):
    """
    Thompson Sampling with Gaussian prior and known precision.
    """

    def __init__(self, p, n_trials=20000, precision=1):
        """
        Args:
            p (list): True means.
            n_trials (int): Trials.
            precision (float): Known precision.
        """
        self.p = p
        self.n_trials = n_trials
        self.k = len(p)

        self.precision = precision

        self.means = np.zeros(self.k)
        self.lambdas = np.ones(self.k)

        self.rewards = []
        self.regrets = []

        self.optimal = max(p)

    def __repr__(self):
        return "ThompsonSampling"

    def pull(self, arm):
        return np.random.normal(self.p[arm], 1)

    def update(self, arm, reward):
        """
        Updates posterior.
        """
        self.lambdas[arm] += self.precision
        self.means[arm] = (
            self.means[arm] * (self.lambdas[arm] - self.precision)
            + self.precision * reward
        ) / self.lambdas[arm]

    def experiment(self):
        """
        Runs Thompson Sampling.
        """
        for _ in range(self.n_trials):

            samples = np.random.normal(
                self.means,
                1 / np.sqrt(self.lambdas)
            )

            arm = np.argmax(samples)

            reward = self.pull(arm)
            self.update(arm, reward)

            regret = self.optimal - self.p[arm]

            self.rewards.append(reward)
            self.regrets.append(regret)

    def report(self):
        df = pd.DataFrame({
            "Bandit": range(self.n_trials),
            "Reward": self.rewards,
            "Algorithm": "ThompsonSampling"
        })

        df.to_csv(os.path.join(RESULTS_DIR, "thompson_sampling.csv"), index=False)
        logger.info(f"Thompson Sampling Total Reward: {sum(self.rewards)}")
        logger.info(f"Thompson Sampling Total Regret: {sum(self.regrets)}")

        return df