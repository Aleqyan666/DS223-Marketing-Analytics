"""
Multi-Armed Bandit Experiment:
- Epsilon-Greedy (ε = 1/t)
- Thompson Sampling (Gaussian with known precision)

Outputs:
- CSV files
- Cumulative reward plots
- Cumulative regret plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# ================= ABSTRACT CLASS =================

class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.
    """

    @abstractmethod
    def __init__(self, p, n_trials=20000):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self, arm):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


# ================= VISUALIZATION =================

class Visualization:
    """
    Handles plotting of rewards and regrets.
    """

    def plot1(self, eg_rewards, ts_rewards):
        """
        Plot cumulative rewards.

        Args:
            eg_rewards (list): Rewards from Epsilon-Greedy
            ts_rewards (list): Rewards from Thompson Sampling
        """
        plt.figure()
        plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
        plt.legend()
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Reward")
        plt.show()

    def plot2(self, eg_regret, ts_regret):
        """
        Plot cumulative regret.

        Args:
            eg_regret (list): Regret from Epsilon-Greedy
            ts_regret (list): Regret from Thompson Sampling
        """
        plt.figure()
        plt.plot(np.cumsum(eg_regret), label="Epsilon-Greedy")
        plt.plot(np.cumsum(ts_regret), label="Thompson Sampling")
        plt.legend()
        plt.title("Cumulative Regret")
        plt.xlabel("Trials")
        plt.ylabel("Regret")
        plt.show()


# ================= EPSILON GREEDY =================

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
        Simulate pulling an arm.

        Args:
            arm (int): Arm index

        Returns:
            float: Reward sample
        """
        return np.random.normal(self.p[arm], 1)

    def update(self, arm, reward):
        """
        Update estimated mean.

        Args:
            arm (int): Selected arm
            reward (float): Observed reward
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        self.values[arm] += (reward - self.values[arm]) / n

    def experiment(self):
        """
        Run experiment with epsilon = 1/t.
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
        Save results and print metrics.
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


# ================= THOMPSON SAMPLING =================

class ThompsonSampling(Bandit):
    """
    Thompson Sampling with Gaussian posterior and known precision.
    """

    def __init__(self, p, n_trials=20000, precision=1):
        """
        Args:
            p (list): True reward means.
            n_trials (int): Number of trials.
            precision (float): Known precision (1/variance)
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
        """
        Sample reward.

        Args:
            arm (int): Arm index

        Returns:
            float: Reward
        """
        return np.random.normal(self.p[arm], 1)

    def update(self, arm, reward):
        """
        Update posterior parameters.

        Args:
            arm (int): Selected arm
            reward (float): Observed reward
        """
        self.lambdas[arm] += self.precision

        self.means[arm] = (
            self.means[arm] * (self.lambdas[arm] - self.precision)
            + self.precision * reward
        ) / self.lambdas[arm]

    def experiment(self):
        """
        Run Thompson Sampling experiment.
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
        """
        Save results and log metrics.
        """
        df = pd.DataFrame({
            "Bandit": range(self.n_trials),
            "Reward": self.rewards,
            "Algorithm": "EpsilonGreedy"
        })

        try:
            df.to_csv(os.path.join(RESULTS_DIR, "epsilon_greedy.csv"), index=False)
            logger.info("Epsilon-Greedy results saved to CSV")
        except Exception as e:
            logger.error(f"Failed to save Epsilon-Greedy CSV: {e}")

        total_reward = sum(self.rewards)
        total_regret = sum(self.regrets)

        logger.info(f"[EpsilonGreedy] Total Reward: {total_reward}")
        logger.warning(f"[EpsilonGreedy] Total Regret: {total_regret}")

        return df

# ================= COMPARISON =================

def comparison():
    """
    Run both algorithms and compare results.
    """
    p = [1, 2, 3, 4]

    eg = EpsilonGreedy(p)
    ts = ThompsonSampling(p)

    logger.info("Running Epsilon-Greedy...")
    eg.experiment()

    logger.info("Running Thompson Sampling...")
    ts.experiment()

    df1 = eg.report()
    df2 = ts.report()

    combined = pd.concat([df1, df2])

    combined.to_csv(os.path.join(RESULTS_DIR, "combined_results.csv"), index=False)    
    vis = Visualization()
    vis.plot1(eg.rewards, ts.rewards)
    vis.plot2(eg.regrets, ts.regrets)


# ================= MAIN =================

if __name__ == '__main__':
    logger.info("Starting Bandit Experiment")

    comparison()

    logger.info("Experiment Completed")