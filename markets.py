import numpy as np


class Market():
    def __init__(self, n):
        self.n = n
        self.q = np.zeros(n)
        self.funds = 0

    def price(self, z: np.array):
        pass

    def trade(self, z: np.array):
        pass


class LMSRMarket(Market):
    def C(self, x: np.array):
        return np.log(np.sum(np.exp(x)))

    def pi(self, x: np.array):
        return np.exp(x) / np.sum(np.exp(x))

    def price(self, z: np.array):
        assert len(z) == self.n, f"Array of length {self.n} expected"
        return self.C(self.q + z) - self.C(self.q)

    def trade(self, z: np.array):
        assert len(z) == self.n, f"Array of length {self.n} expected"
        assert (
            self.q + z >= 0).all(), f"Cannot sell more shares than what is currently on the market"
        self.funds = self.funds + (p := self.price(z))
        self.q = self.q + z
        return p


class Trader():
    def __init__(self, funds):
        self.funds = funds

    def trade(self, M: Market, z: np.array):
        assert M.price(
            z) <= self.funds, "Insufficient funds to make this trade"
        self.funds = self.funds - M.trade(z)

    def step(self, M: Market):
        pass


class UnitTrader(Trader):
    def step(self, M: Market):
        pass



if __name__ == "__main__":
    pass