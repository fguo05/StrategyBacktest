import pandas as pd

from strategy import Strategy
from utils import get_config


class Backtest:
    def __init__(self, balance:float, position:float, strategy:Strategy, signals:list, market_data:pd.DataFrame, funding_rate:float=None):
        self.balance = balance      # 账户余额
        self.position = position    # 仓位/合约
        self.strategy = strategy    # 策略
        self.signals = signals      # 信号
        self.market_data = market_data  # 市场价格数据
        self.funding_rate = funding_rate if funding_rate else get_config()["funding_rate"]  # 资金费率

    def backtest(self):
        pass