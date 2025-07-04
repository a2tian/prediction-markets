from markets import *


def two_market_sim(traders, m1, m2, m1_given_m2):

    M1, M2 = LMSRMarket(2), LMSRMarket(2)

    for _ in range(500):
        j = np.random.randint(len(traders))
        traders[j].step([M1, M2])

    plt.scatter(range(len(M1.history)), np.array(M1.history)[:, 0])
    plt.scatter(range(len(M2.history)), np.array(M2.history)[:, 0])
    plt.show()


if __name__ == "__main__":
    m1_ground = 0.4
    m2_ground = 0.6
    m1_given_m2 = 0.3
    m2_given_m1 = m1_given_m2 * m2_ground / m1_ground  # Bayes' rule
    n_traders = 100
    traders1 = [random_binary_market_trader(
        math.inf, [m1_ground, m2_ground]) for _ in range(n_traders)]
    two_market_sim(traders1, m1_ground, m2_ground, m1_given_m2)

    traders2 = traders1 + [SimpleAutoTrader(math.inf, lambda A: unit_trade(
        A, [[(r := m1_given_m2 * A[1].current_prices()[0]), 1-r], A[1].current_prices()])) for _ in range(n_traders)] + [SimpleAutoTrader(math.inf, lambda A: unit_trade(
        A, [A[0].current_prices(), [(r := m2_given_m1 * A[0].current_prices()[0]), 1-r]])) for _ in range(n_traders)]

    two_market_sim(traders2, m1_ground, m2_ground, m1_given_m2)