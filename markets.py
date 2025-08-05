import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections.abc import Callable


class Market():
    def __init__(self, n):
        self.n = n
        self.q = np.zeros(n)
        self.trades = np.zeros(n)
        self.funds = 0
        self.history = [self.current_prices()]

    def current_prices(self):
        pass

    def price(self, z: np.array):
        pass

    def trade(self, z: np.array):
        pass
    


class LMSRMarket(Market):
    def C(self, x: np.array):
        return np.log(np.sum(np.exp(x)))

    def pi(self, x: np.array):
        return np.exp(x) / np.sum(np.exp(x))

    def current_prices(self):
        return self.pi(self.q)

    def price(self, z: np.array):
        assert len(z) == self.n, f"Array of length {self.n} expected"
        return self.C(self.q + z) - self.C(self.q)

    def trade(self, z: np.array):
        assert len(z) == self.n, f"Array of length {self.n} expected"
        assert (
            self.q + z >= 0).all(), f"Cannot sell more shares than what is currently on the market"
        self.funds = self.funds + (p := self.price(z))
        self.q += z
        self.trades += z
        self.history.append(self.current_prices())
        return p


class Trader():
    def __init__(self, funds: float):
        self.funds = funds
        self.inventory = dict()

    def trade(self, M: Market, z: np.array):
        if not M in self.inventory:
            self.inventory[M] = np.zeros(M.n)
        assert (
            self.inventory[M] + z >= 0).all(), f"Cannot sell more shares than what is currently owned"
        assert M.price(
            z) <= self.funds, f"Insufficient funds to make this trade"
        self.funds = self.funds - M.trade(z)
        self.inventory[M] += z


class SimpleAutoTrader(Trader):
    def __init__(self, funds, rule):
        super().__init__(funds)
        self.rule = rule

    def step(self, markets: List[Market]):
        for i, z in enumerate(self.rule(markets)):
            self.trade(markets[i], z)

class RandomMultiBernoulliTrader(Trader):
    def __init__(self, funds, ground, idx):
        super().__init__(funds)
        belief = np.random.beta(100 * ground, 100 * (1 - ground))
        self.belief = np.array([belief, 1 - belief])
        self.idx = idx

    def step(self, markets: List[Market]):
        z = (self.belief > markets[self.idx].current_prices()).astype(int) * 0.01
        self.trade(markets[self.idx], z)



def unit_trade(markets, beliefs):
    return [(M.current_prices() < b).astype(int) * 0.01 for M, b in zip(markets, beliefs)]


def random_binary_belief(grounds, n=100):
    rs = [np.random.beta(n * ground, n * (1 - ground)) for ground in grounds]
    return [np.array([r, 1 - r]) for r in rs]


def random_binary_market_trader(funds, ground, n=100):
    return SimpleAutoTrader(funds, lambda A: unit_trade(A, random_binary_belief(ground, n)))


if __name__ == "__main__":
    M = LMSRMarket(2)
    ground = 0.67
    n_traders = 100
    traders = [random_binary_market_trader(
        math.inf, [ground]) for _ in range(n_traders)]

    prices = [M.current_prices()[0]]
    for _ in range(200):
        j = np.random.randint(0, n_traders)
        traders[j].step([M])
        prices.append(M.current_prices()[0])

    plt.scatter(list(range(0, 201)), prices)
    plt.show()
