# search_engine.py
"""
回测运行脚本，集成了 utils 中的信号生成和数据下载功能
支持自动下载 OKX 价格数据、生成信号并执行回测
"""
import pandas as pd
import os
from datetime import datetime, timedelta
from backtest import Simulator
from config import BACKTEST_CONFIG
import utils

def ensure_price_data(symbol: str, start_date: str, end_date: str, timeframe: str = '15m'):
    """
    确保价格数据存在，若不存在则调用 utils.get_okx_data 下载
    返回 DataFrame (index: timestamp, columns: ['open','high','low','close','volume'])
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    # OKX 数据起始时间需精确到秒，这里使用当天 00:00:00Z
    start_str = f"{start_date}T00:00:00Z"
    filename = f"okx_{symbol}_{start_str.replace(':', '-')}_{timeframe}.csv"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"下载 {symbol} 价格数据（{start_date} 至 {end_date}）...")
        # 注意：get_okx_data 会自动保存 CSV，并返回 DataFrame
        df = utils.get_okx_data(symbol, start_str, timeframe, asset_type="perp")
    else:
        print(f"已存在 {symbol} 价格数据文件: {filepath}")
        df = utils.load_ohlcv_csv(filepath)

    # 截取回测区间内的数据（get_okx_data 可能返回从 start 到当前的所有数据）
    df = df.loc[start_date:end_date]
    return df

def ensure_signal_data(conn, symbols: list, start_date: str, end_date: str, signal_type: str = 'title'):
    """
    确保信号数据存在，若不存在则调用 utils.get_daily_signal 生成
    返回字典 {symbol: DataFrame(index=date, columns=['signal'])}
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    signal_file = os.path.join(data_dir, f"signals_{signal_type}_{start_date}_{end_date}.csv")

    if os.path.exists(signal_file):
        print(f"加载已有信号文件: {signal_file}")
        signals_df = pd.read_csv(signal_file, parse_dates=['date'])
    else:
        print(f"生成信号数据（{signal_type}），这可能较慢，请耐心等待...")
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current = start
        records = []

        while current <= end:
            cur_date_str = current.strftime('%Y-%m-%d')
            for sym in symbols:
                print(f"  处理 {cur_date_str} {sym}...")
                signal_dict = utils.get_daily_signal(conn, sym, cur_date_str)

                if signal_dict is None:
                    # 新闻数量不足，当日无信号 -> 默认不动 (0)
                    signal = 0
                else:
                    if signal_type == 'title':
                        signal = signal_dict['title']['signal']
                    elif signal_type == 'title_content':
                        signal = signal_dict['title_content']['signal']
                    else:
                        raise ValueError("signal_type 必须是 'title' 或 'title_content'")
                    if signal is None:
                        signal = 0  # 分数缺失时也视为不动
                records.append({
                    'date': cur_date_str,
                    'symbol': sym,
                    'signal': signal
                })
            current += timedelta(days=1)

        signals_df = pd.DataFrame(records)
        signals_df.to_csv(signal_file, index=False)
        print(f"信号已保存至: {signal_file}")

    # 转换为回测所需的字典格式
    signal_dict = {}
    for sym in symbols:
        df_sym = signals_df[signals_df['symbol'] == sym].set_index('date')[['signal']]
        df_sym.index = pd.to_datetime(df_sym.index)  # 确保日期类型
        signal_dict[sym] = df_sym
    return signal_dict

def backtestengine():
    # 1. 加载配置
    config = BACKTEST_CONFIG.copy()
    symbols = config['symbols']
    start_date = config['start_date']
    end_date = config['end_date']
    timeframe = config.get('interval', '15m')
    signal_type = config.get('signal_type', 'title')  # 可选 'title' 或 'title_content'

    # 2. 确保价格数据存在
    price_data = {}
    for sym in symbols:
        price_data[sym] = ensure_price_data(sym, start_date, end_date, timeframe)

    # 3. 确保信号数据存在（需要数据库连接）
    conn = utils.create_db_connection('test')  # 可根据需要改为 'real'
    if conn is None:
        print("数据库连接失败，无法生成信号。")
        return
    try:
        signal_dict = ensure_signal_data(conn, symbols, start_date, end_date, signal_type)
    finally:
        conn.close()

    # 4. 运行回测
    print("\n开始回测...")
    sim = Simulator(config, price_data=price_data, signal_data=signal_dict)
    sim.run()

    # 5. 输出绩效指标
    metrics = sim.calculate_metrics()
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")

    # 6. 保存结果
    net_series = sim.get_net_value_series()
    net_series.to_csv('net_value.csv')
    trades_df = sim.get_trade_records_df()
    trades_df.to_csv('trade_records.csv', index=False)
    print("\n净值曲线已保存至 net_value.csv，交易记录已保存至 trade_records.csv")

    # 7. 可选：绘制净值曲线
    try:
        import matplotlib.pyplot as plt
        net_series.plot(figsize=(12,5), title='Net Value Curve')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib 未安装，跳过绘图。")

if __name__ == "__main__":
    backtestengine()