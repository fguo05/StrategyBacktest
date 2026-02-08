import pandas as pd

from strategy import Strategy
from utils import get_config


class Backtest:
    def __init__(self, position:float, strategy:Strategy, signals:list, market_data:pd.DataFrame, funding_rate:float=None):
        self.position = position
        self.strategy = strategy
        self.signals = signals
        self.market_data = market_data
        self.funding_rate = funding_rate if funding_rate else get_config()["funding_rate"]

    def backtest(self):
        pass