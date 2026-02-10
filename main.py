from backtest import Backtest
from utils import *


# data = get_okx_data("BTC", '2022-01-01T00:00:00Z', "15m")
# print(data.head())
# print(data.tail())

conn = create_db_connection("test")
result = get_daily_signal(conn, "BTC", "2025-01-02")
print(result)
