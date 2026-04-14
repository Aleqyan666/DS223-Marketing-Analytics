from epsilon_greedy import EpsilonGreedy
from thompson_sampling import ThompsonSampling
from visualization import Visualization
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    """
    Runs full experiment and comparison.
    """

    p = [1, 2, 3, 4]

    eg = EpsilonGreedy(p)
    ts = ThompsonSampling(p)

    eg.experiment()
    ts.experiment()

    df1 = eg.report()
    df2 = ts.report()

    combined = pd.concat([df1, df2])
    combined.to_csv(os.path.join(RESULTS_DIR, "combined_results.csv"), index=False)
    vis = Visualization()
    vis.plot_rewards(eg.rewards, ts.rewards)
    vis.plot_regret(eg.regrets, ts.regrets)

if __name__ == "__main__":
    main()