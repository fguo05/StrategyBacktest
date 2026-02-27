# backtest/部分完整代码
# position
"""
账户持仓状态管理模块
记录每个币种的持仓信息，并计算未实现盈亏
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class Position:
    """单个币种的持仓数据"""
    symbol: str
    direction: Optional[str] = None          # 'long' 或 'short' 或 None
    quantity: float = 0.0                     # 持仓数量（正数）
    entry_price: float = 0.0                   # 开仓均价
    entry_time: Optional[datetime] = None      # 开仓时间
    stop_loss: float = 0.0                      # 止损价
    take_profit: float = 0.0                    # 止盈价

    def is_open(self) -> bool:
        """是否持有仓位"""
        return self.direction is not None and self.quantity > 0

    def unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏（绝对金额）"""
        if not self.is_open():
            return 0.0
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """计算未实现盈亏百分比（相对于开仓名义本金）"""
        if not self.is_open() or self.entry_price == 0:
            return 0.0
        if self.direction == 'long':
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def close(self):
        """平仓，重置持仓"""
        self.direction = None
        self.quantity = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0


# execution
"""
交易执行模块
根据信号进行开平仓操作，更新持仓和交易记录
"""

@dataclass
class TradeRecord:
    """单笔交易记录"""
    symbol: str
    direction: str          # 'long' or 'short'
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float              # 盈亏（绝对值）
    pnl_pct: float          # 盈亏百分比
    reason: str             # 平仓原因：'signal', 'stop_loss', 'take_profit', 'end_of_backtest'


class Execution:
    """
    执行模块，负责处理信号和账户操作
    """
    def __init__(self, config: dict):
        self.config = config
        self.symbols = config['symbols']
        self.leverage = config.get('leverage', {})           # 币种杠杆倍数
        self.stop_loss_pct = config.get('stop_loss_pct', 0.01)
        self.take_profit_pct = config.get('take_profit_pct', 0.01)
        self.initial_capital = config['initial_capital']
        self.capital_allocation = config.get('capital_allocation', {})

        # 初始化持仓字典 {symbol: Position}
        self.positions: Dict[str, Position] = {}
        for sym in self.symbols:
            self.positions[sym] = Position(symbol=sym)

        # 交易记录列表
        self.trade_records: List[TradeRecord] = []

        # 各币种已实现盈亏累计（用于计算净值）
        self.realized_pnl: Dict[str, float] = {sym: 0.0 for sym in self.symbols}

        # 连续亏损计数器
        self.consecutive_losses = 0
        self.consecutive_loss_limit = config.get('consecutive_loss_limit', 4)
        self.pause_days = config.get('pause_days', 7)
        self.pause_until: Optional[datetime.date] = None

    def _calculate_quantity(self, symbol: str, price: float, current_net: float) -> float:
        """根据当前净值和杠杆计算开仓数量"""
        lev = self.leverage.get(symbol, 1)
        return current_net * lev / price

    def _open_position(self, symbol: str, direction: str, price: float, time: datetime, current_net: float):
        """开仓，设置止损止盈价"""
        print(f"[DEBUG] _open_position: symbol={symbol}, direction={direction}, price={price}, current_net={current_net}")
        pos = self.positions[symbol]
        if pos.is_open():
            # 如果已有反向仓位，应先平仓（调用者保证）
            return

        quantity = self._calculate_quantity(symbol, price, current_net)
        print(f"[DEBUG] calculated quantity: {quantity}")
        pos.direction = direction
        pos.quantity = quantity
        pos.entry_price = price
        pos.entry_time = time

        # 设置止盈止损价
        if direction == 'long':
            pos.stop_loss = price * (1 - self.stop_loss_pct)
            pos.take_profit = price * (1 + self.take_profit_pct)
        else:  # short
            pos.stop_loss = price * (1 + self.stop_loss_pct)
            pos.take_profit = price * (1 - self.take_profit_pct)

    def _close_position(self, symbol: str, price: float, time: datetime, reason: str):
        """平仓，记录交易，更新已实现盈亏和连续亏损"""
        pos = self.positions[symbol]
        if not pos.is_open():
            return

        # 计算盈亏
        if pos.direction == 'long':
            pnl = (price - pos.entry_price) * pos.quantity
            pnl_pct = (price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - price) * pos.quantity
            pnl_pct = (pos.entry_price - price) / pos.entry_price

        # 记录交易
        trade = TradeRecord(
            symbol=symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=time,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason
        )
        self.trade_records.append(trade)

        # 更新已实现盈亏
        self.realized_pnl[symbol] += pnl

        # 更新连续亏损
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.consecutive_loss_limit:
                self.pause_until = time.date() + timedelta(days=self.pause_days)
        else:
            self.consecutive_losses = 0

        # 清空持仓
        pos.close()

    def process_signal(self, symbol: str, signal: int, price: float, time: datetime, current_net: float):
        """
        根据信号处理交易
        signal: 2-开多, 1-不动, 0-清仓, -1-不动, -2-开空
        current_net: 当前该币种净值（用于开仓数量计算）
        """
        print(f"[DEBUG] process_signal: symbol={symbol}, signal={signal}, price={price}, current_net={current_net}, is_open={self.positions[symbol].is_open()}")
        if self.pause_until and time.date() <= self.pause_until:
            # 暂停交易期间不执行信号
            return

        pos = self.positions[symbol]

        # 处理清仓信号 (0)
        if signal == 0:
            if pos.is_open():
                self._close_position(symbol, price, time, 'signal')
            # 清仓后不再进行任何开仓操作
            return

        # 处理不动信号 (1 和 -1)
        if signal == 1 or signal == -1:
            return

        # 处理开仓信号 (2 和 -2)
        if signal == 2:           # 开多
            if pos.is_open() and pos.direction == 'short':
                self._close_position(symbol, price, time, 'signal')
            if not pos.is_open():
                self._open_position(symbol, 'long', price, time, current_net)
        elif signal == -2:        # 开空
            if pos.is_open() and pos.direction == 'long':
                self._close_position(symbol, price, time, 'signal')
            if not pos.is_open():
                self._open_position(symbol, 'short', price, time, current_net)
                
    def check_stop_loss_take_profit(self, symbol: str, price: float, time: datetime):
        """检查单个币种是否触发止盈止损"""
        pos = self.positions[symbol]
        if not pos.is_open():
            return
        if pos.direction == 'long':
            if price <= pos.stop_loss:
                self._close_position(symbol, price, time, 'stop_loss')
            elif price >= pos.take_profit:
                self._close_position(symbol, price, time, 'take_profit')
        else:  # short
            if price >= pos.stop_loss:
                self._close_position(symbol, price, time, 'stop_loss')
            elif price <= pos.take_profit:
                self._close_position(symbol, price, time, 'take_profit')


# simulator
"""
回测模拟器：时间推进 + 状态更新
"""

class Simulator:
    """
    回测模拟器，负责时间循环和净值记录
    """
    def __init__(self, config: dict, price_data: Dict[str, pd.DataFrame], signal_data: Dict[str, pd.DataFrame]):
        """
        config: 配置字典
        price_data: {symbol: DataFrame with columns ['close'] and datetime index}
        signal_data: {symbol: DataFrame with columns ['signal'] and date index}
        """
        self.config = config
        self.price_data = price_data
        self.signal_data = signal_data
        self.symbols = config['symbols']
        self.initial_capital = config['initial_capital']
        self.capital_allocation = config.get('capital_allocation', {})
        self.trade_time = config['trade_time']          # 如 '11:00' (UTC)
        self.interval = config['interval']              # 如 '15min'
        self.start_date = pd.Timestamp(config['start_date'], tz='UTC')
        self.end_date = pd.Timestamp(config['end_date'], tz='UTC')
        print(f"Initial capital: {self.initial_capital}")
        print(f"Capital allocation: {self.capital_allocation}")

        # 初始化执行模块
        self.execution = Execution(config)

        # 记录历史净值（时间，总净值）
        self.net_value_history: List[tuple] = []

        # 预计算所有时间点
        self._prepare_time_points()

    def _prepare_time_points(self):
        """生成回测区间内所有interval时间点（UTC）"""
        all_times = []
        for sym in self.symbols:
            all_times.append(self.price_data[sym].index)

        if not all_times:
            self.time_points = pd.DatetimeIndex([])
            return

        # 合并所有时间索引
        combined = all_times[0]
        for idx in all_times[1:]:
            combined = combined.union(idx)

        # 根据回测起止时间筛选
        mask = (combined >= self.start_date) & (combined <= self.end_date)
        self.time_points = combined[mask]

    def _get_price(self, symbol: str, time: datetime) -> float:
        """获取指定时间的收盘价（向前填充）"""
        df = self.price_data[symbol]
        if time in df.index:
            return df.loc[time, 'close']
        else:
            # 使用前一个有效价格
            idx = df.index.get_indexer([time], method='ffill')
            if idx[0] >= 0:
                return df.iloc[idx[0]]['close']
            else:
                raise ValueError(f"No price for {symbol} at {time}")

    def _calculate_net_value(self, time: datetime) -> float:
        """计算当前总净值 = 初始资本 + 所有已实现盈亏 + 所有未实现盈亏"""
        total = self.initial_capital
        # 加上已实现盈亏
        for sym in self.symbols:
            total += self.execution.realized_pnl.get(sym, 0.0)
        # 加上未实现盈亏
        for sym in self.symbols:
            pos = self.execution.positions[sym]
            if pos.is_open():
                price = self._get_price(sym, time)
                total += pos.unrealized_pnl(price)
        return total

    def _get_symbol_net_value(self, symbol: str, time: datetime) -> float:
        """计算单个币种当前净值（用于开仓数量计算）"""
        # 该币种净值 = 初始分配 + 该币种已实现盈亏 + 该币种未实现盈亏
        alloc = self.capital_allocation.get(symbol, 1.0 / len(self.symbols))
        base = self.initial_capital * alloc
        realized = self.execution.realized_pnl.get(symbol, 0.0)
        pos = self.execution.positions[symbol]
        print(f"[DEBUG] _get_symbol_net_value: {symbol}, alloc={alloc}, base={base}, realized={realized}")
        if pos.is_open():
            price = self._get_price(symbol, time)
            unrealized = pos.unrealized_pnl(price)
        else:
            unrealized = 0.0
        return base + realized + unrealized

    def run(self):
        """运行回测主循环"""
        prev_date = None
        for time in self.time_points:
            current_date = time.date()

            # 1. 检查止盈止损（对所有币种）
            for sym in self.symbols:
                price = self._get_price(sym, time)
                self.execution.check_stop_loss_take_profit(sym, price, time)

            # 2. 如果是交易时间且新的一天，执行信号交易
            trade_hour, trade_minute = map(int, self.trade_time.split(':'))
            if time.hour == trade_hour and time.minute == trade_minute and current_date != prev_date:
                print(f"Executing signal at {time}")
                # 获取当日信号
                for sym in self.symbols:
                    # 从signal_data中获取当天信号，若没有则默认为0
                    df = self.signal_data[sym]
                    # 将索引转换为 datetime.date 对象（处理字符串或 Timestamp）
                    df.index = pd.to_datetime(df.index).date
                    if current_date in df.index:
                        signal = df.loc[current_date, 'signal']
                    else:
                        signal = 0
                    price = self._get_price(sym, time)
                    sym_net = self._get_symbol_net_value(sym, time)
                    self.execution.process_signal(sym, signal, price, time, sym_net)
                prev_date = current_date

            # 3. 记录净值（可选：每个时间点记录或每天记录）
            # 这里简单地在每个时间点记录，以便后续分析
            total_net = self._calculate_net_value(time)
            self.net_value_history.append((time, total_net))

        # 回测结束，平掉所有持仓（按最后价格）
        final_time = self.time_points[-1]
        for sym in self.symbols:
            if self.execution.positions[sym].is_open():
                price = self._get_price(sym, final_time)
                self.execution._close_position(sym, price, final_time, 'end_of_backtest')
        # 记录最终净值
        final_net = self._calculate_net_value(final_time)
        self.net_value_history.append((final_time, final_net))

    def get_net_value_series(self) -> pd.Series:
        """返回净值时间序列"""
        df = pd.DataFrame(self.net_value_history, columns=['time', 'net_value'])
        df.set_index('time', inplace=True)
        return df['net_value']

    def get_trade_records_df(self) -> pd.DataFrame:
        """返回交易记录DataFrame"""
        return pd.DataFrame([vars(t) for t in self.execution.trade_records])

    def calculate_metrics(self) -> dict:
        """计算常用绩效指标"""
        net_series = self.get_net_value_series()
        if len(net_series) < 2:
            return {}

        # 日收益率（按日重采样）
        daily_net = net_series.resample('D').last().dropna()
        daily_returns = daily_net.pct_change().dropna()

        total_return = (daily_net.iloc[-1] - self.initial_capital) / self.initial_capital
        days = (daily_net.index[-1] - daily_net.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0.0

        # 最大回撤
        rolling_max = daily_net.expanding().max()
        drawdown = (daily_net - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 夏普比率（假设无风险利率0）
        if len(daily_returns) > 0 and daily_returns.std() != 0:
            sharpe = np.sqrt(365) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0.0

        # 胜率
        trades = self.get_trade_records_df()
        win_rate = (trades['pnl'] > 0).mean() if len(trades) > 0 else 0.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': len(trades)
        }