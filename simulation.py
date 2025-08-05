from markets import LMSRMarket, RandomMultiBernoulliTrader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import math

def bernoulli_conditionals(p1: float, p2: float, rho: float):
    """
    Parameters
    ----------
    p1   : P(A = 1)
    p2   : P(B = 1)
    rho  : desired Pearson correlation between A and B

    Returns
    -------
    P_A_given_B : 2×2 ndarray
        [[P(A=1 | B=1), P(A=1 | B=0)],
         [P(A=0 | B=1), P(A=0 | B=0)]]

    P_B_given_A : 2×2 ndarray
        [[P(B=1 | A=1), P(B=1 | A=0)],
         [P(B=0 | A=1), P(B=0 | A=0)]]
    """
    # --- feasibility check --------------------------------------------------
    s1, s2 = np.sqrt(p1 * (1 - p1)), np.sqrt(p2 * (1 - p2))
    rho_min = (max(0.0, p1 + p2 - 1) - p1 * p2) / (s1 * s2)
    rho_max = (min(p1, p2)            - p1 * p2) / (s1 * s2)
    if not (rho_min - 1e-12 <= rho <= rho_max + 1e-12):
        raise ValueError(f"ρ must lie in [{rho_min:.3g}, {rho_max:.3g}] for "
                         f"these marginals.")

    # --- joint probabilities ------------------------------------------------
    p11 = rho * s1 * s2 + p1 * p2
    p10 = p1 - p11          # A=1, B=0
    p01 = p2 - p11          # A=0, B=1
    p00 = 1 - p11 - p10 - p01

    # --- conditional matrices ----------------------------------------------
    # P(A | B)
    P_A_given_B = np.array([
        [p11 / p2,           p10 / (1 - p2)],     # A = 1
        [p01 / p2,           p00 / (1 - p2)]      # A = 0
    ])

    # P(B | A)
    P_B_given_A = np.array([
        [p11 / p1,           p01 / (1 - p1)],     # B = 1
        [p10 / p1,           p00 / (1 - p1)]      # B = 0
    ])

    return P_B_given_A, P_A_given_B


def round(traders, A, B, *args):
    random.shuffle(traders)
    for trader in traders:
        trader.step([A, B])

def round_with_update(traders, A, B, Ua, Ub):
    random.shuffle(traders)
    for trader in traders:
        trader.step([A, B])
        va = np.log(Ub @ B.current_prices())
        vb = np.log(Ua @ A.current_prices())
        va, vb = va + np.min(va) + 10, vb + np.min(vb) + 10  # allow selling
        A.q, B.q = va, vb

def simulate(p1: float, p2: float, rho: float, update=True):
    A, B = LMSRMarket(2), LMSRMarket(2)
    Ua, Ub = bernoulli_conditionals(p1, p2, rho)
    traders = [RandomMultiBernoulliTrader(math.inf, p1, 0)] * 50 + \
              [RandomMultiBernoulliTrader(math.inf, p2, 1)] * 50
    
    for _ in range(1):
        if update:
            round_with_update(traders, A, B, Ua, Ub)
        else:
            round(traders, A, B)

    prices_A = np.array(A.history)[:, 0]
    prices_B = np.array(B.history)[:, 0]

    # Create a DataFrame for plotting
    df = pd.DataFrame({"A": prices_A, "B": prices_B})
    df = df.reset_index(names="Step")
    df = pd.melt(df, id_vars=["Step"], value_vars=["A", "B"], var_name="Market", value_name="Price")

    # Average change in price
    delta_A, delta_B = np.diff(prices_A).mean(), np.diff(prices_B).mean()
    loss_A, loss_B = A.trades @ np.array([p1, 1 - p1]), B.trades @ np.array([p2, 1 - p2])

    # sns.scatterplot(data=np.array(A.history)[:, 0], label='A')
    # sns.scatterplot(data=np.array(B.history)[:, 0], label='B')
    # # plt.plot(range(len(A.history)), np.array(A.history)[:, 0], label='A')
    # # plt.plot(range(len(B.history)), np.array(B.history)[:, 0], label='B')
    # plt.xlabel('Steps')
    # plt.ylabel('Price')
    # plt.title(f'p1={p1}, p2={p2}, ρ={rho}')
    # plt.legend(loc="center right")
    # plt.show()
    return df, delta_A, delta_B, loss_A, loss_B


if __name__ == "__main__":
    # Example parameters
    p1 = 0.6
    p2 = 0.4
    rho = -0.8  # Desired correlation

    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.8)

    history = pd.DataFrame(columns=["rho", "delta_A", "delta_B", "loss_A", "loss_B"])
    for rho in np.arange(-1, 0.66, 0.05):
        for trial in range(10):
            df, delta_A, delta_B, loss_A, loss_B = simulate(p1, p2, rho, update=False)
            history = pd.concat([history, pd.DataFrame({"rho": [rho],
                                          "delta_A": [delta_A],
                                          "delta_B": [delta_B],
                                          "loss_A": [loss_A],
                                          "loss_B": [loss_B]})]).reset_index(drop=True)

    # Plotting the results
    plt.figure(figsize=(12, 9))
    sns.lineplot(data=history, x="rho", y="delta_A", label="A", marker="o")
    sns.lineplot(data=history, x="rho", y="delta_B", label="B", marker="o")
    plt.xlabel("Correlation (ρ)")
    plt.ylabel("Average Price Change")
    plt.title("Average Price Change vs Correlation, LMSR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("liq_no_mod.png")


    history = pd.DataFrame(columns=["rho", "delta_A", "delta_B", "loss_A", "loss_B"])
    for rho in np.arange(-1, 0.66, 0.05):
        for trial in range(10):
            df, delta_A, delta_B, loss_A, loss_B = simulate(p1, p2, rho, update=True)
            history = pd.concat([history, pd.DataFrame({"rho": [rho],
                                          "delta_A": [delta_A],
                                          "delta_B": [delta_B],
                                          "loss_A": [loss_A],
                                          "loss_B": [loss_B]})]).reset_index(drop=True)

    plt.figure(figsize=(12, 9))
    sns.lineplot(data=history, x="rho", y="delta_A", label="A", marker="o")
    sns.lineplot(data=history, x="rho", y="delta_B", label="B", marker="o")
    plt.xlabel("Correlation (ρ)")
    plt.ylabel("Average Price Change")
    plt.title("Average Price Change vs Correlation, Our Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("liq_mod.png")


    history = pd.DataFrame(columns=["rho", "delta_A", "delta_B", "loss_A", "loss_B"])
    for rho in np.arange(-1, 0.66, 0.05):
        for trial in range(10):
            df, delta_A, delta_B, loss_A, loss_B = simulate(p1, p2, rho, update=False)
            history = pd.concat([history, pd.DataFrame({"rho": [rho],
                                          "delta_A": [delta_A],
                                          "delta_B": [delta_B],
                                          "loss_A": [loss_A],
                                          "loss_B": [loss_B]})]).reset_index(drop=True)

    # Plotting the results
    plt.figure(figsize=(12, 9))
    sns.lineplot(data=history, x="rho", y="loss_A", label="A", marker="o")
    sns.lineplot(data=history, x="rho", y="loss_B", label="B", marker="o")
    plt.xlabel("Correlation (ρ)")
    plt.ylabel("Exdpected Loss")
    plt.title("Expected Loss vs Correlation, LMSR")
    plt.legend()
    plt.ylim(0.15, 0.35)
    plt.tight_layout()
    plt.savefig("loss_no_mod.png")


    history = pd.DataFrame(columns=["rho", "delta_A", "delta_B", "loss_A", "loss_B"])
    for rho in np.arange(-1, 0.66, 0.05):
        for trial in range(10):
            df, delta_A, delta_B, loss_A, loss_B = simulate(p1, p2, rho, update=True)
            history = pd.concat([history, pd.DataFrame({"rho": [rho],
                                          "delta_A": [delta_A],
                                          "delta_B": [delta_B],
                                          "loss_A": [loss_A],
                                          "loss_B": [loss_B]})]).reset_index(drop=True)

    plt.figure(figsize=(12, 9))
    sns.lineplot(data=history, x="rho", y="loss_A", label="A", marker="o")
    sns.lineplot(data=history, x="rho", y="loss_B", label="B", marker="o")
    plt.xlabel("Correlation (ρ)")
    plt.ylabel("Expected Loss")
    plt.title("Expected Loss vs Correlation, Our Model")
    plt.legend()
    plt.ylim(0.15, 0.35)
    plt.tight_layout()
    plt.savefig("loss_mod.png")

