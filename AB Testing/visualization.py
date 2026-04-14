import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """
    Handles plotting of rewards and regret.
    """

    def plot_rewards(self, eg_rewards, ts_rewards):
        plt.figure()
        plt.plot(np.cumsum(eg_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
        plt.legend()
        plt.title("Cumulative Rewards")
        plt.show()

    def plot_regret(self, eg_regret, ts_regret):
        plt.figure()
        plt.plot(np.cumsum(eg_regret), label="Epsilon-Greedy")
        plt.plot(np.cumsum(ts_regret), label="Thompson Sampling")
        plt.legend()
        plt.title("Cumulative Regret")
        plt.show()