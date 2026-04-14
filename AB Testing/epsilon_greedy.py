import numpy as np
import pandas as pd
from bandit_base import Bandit
from loguru import logger
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm with decaying epsilon (1/t).
    """

    def __init__(self, p, n_trials=20000):
        """
        Args:
            p (list): True reward means.
            n_trials (int): Number of trials.
        """
        self.p = p
        self.n_trials = n_trials
        self.k = len(p)

        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)

        self.rewards = []
        self.regrets = []

        self.optimal = max(p)

    def __repr__(self):
        return "EpsilonGreedy"

    def pull(self, arm):
        """
        Returns sampled reward.
        """
        return np.random.normal(self.p[arm], 1)

    def update(self, arm, reward):
        """
        Updates estimated mean.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def experiment(self):
        """
        Runs experiment using epsilon = 1/t.
        """
        for t in range(1, self.n_trials + 1):

            epsilon = 1 / t

            if np.random.random() < epsilon:
                arm = np.random.randint(self.k)
            else:
                arm = np.argmax(self.values)

            reward = self.pull(arm)
            self.update(arm, reward)

            regret = self.optimal - self.p[arm]

            self.rewards.append(reward)
            self.regrets.append(regret)

    def report(self):
        """
        Saves results and prints metrics.
        """
        df = pd.DataFrame({
            "Bandit": range(self.n_trials),
            "Reward": self.rewards,
            "Algorithm": "EpsilonGreedy"
        })

        df.to_csv(os.path.join(RESULTS_DIR, "epsilon_greedy.csv"), index=False)

        logger.info(f"Epsilon-Greedy Total Reward: {sum(self.rewards)}")
        logger.info(f"Epsilon-Greedy Total Regret: {sum(self.regrets)}")

        return df