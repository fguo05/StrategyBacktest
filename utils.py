import os

import pymysql
import requests
from datetime import datetime, timedelta, timezone

import yfinance as yf
import ccxt
import pandas as pd
import openai

from config import *


_config = None # 配置全局变量，不要直接用DEFAULT_CONFIG

def get_config():
    global _config
    if _config is None:
        _config = DEFAULT_CONFIG.copy()
    return _config.copy()


def create_db_connection(db_name):
    """
    建立数据库连接
    :param db_name: "real"实盘数据库，"test"测试数据库
    :return: Connection
    """
    if db_name not in ["real", "test"]:
        print("数据库名错误！")
        return None

    return pymysql.connect(
        host='rm-uf6q5h4a7tkthf82cno.mysql.rds.aliyuncs.com',  # 公网地址
        port=3306,  # 端口
        user='db_USER1',  # 数据库账号
        password='Cangjie!2025',  # 数据库密码
        db='db_test1' if db_name == "real" else "db_agent_test",
        charset='utf8mb4',  # 字符编码
    )


def get_alphavantage_news(ticker: str, cur_date:str=None, topics: str = None) -> list:
    """cur_date为空则默认以今天为 to_time 获取24h新闻"""

    if ticker in ALPHAVANTAGE_TICKER_MAP:
        ticker = ALPHAVANTAGE_TICKER_MAP[ticker]

    if cur_date:
        cur_date_obj = datetime.strptime(cur_date, '%Y-%m-%d')
        start_time = cur_date_obj
        from_time = start_time.strftime("%Y%m%dT%H%M")
        to_time = (cur_date_obj + timedelta(hours=24)).strftime("%Y%m%dT%H%M")
    else:
        cur_date_obj = datetime.now(timezone.utc)
        start_time = cur_date_obj - timedelta(hours=24)
        from_time = start_time.strftime("%Y%m%dT%H%M")
        to_time = cur_date_obj.strftime("%Y%m%dT%H%M")

    # 构造API请求
    api_key = 'NJ7FJN6R0LIM0W77'
    base_url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
    url = f"{base_url}&tickers={ticker}&time_from={from_time}&time_to={to_time}&apikey={api_key}"

    response = requests.get(url)
    data = response.json()

    if "feed" not in data:
        print(f"{ticker} 在过去24小时内没有找到相关新闻")
        return []

    news_list = [
        {
            "title": news.get('title', '').strip(),
            "content": news.get('summary', '').strip(),
            "publish_time": news.get('time_published', ''),
            "source": news.get('source', '').strip(),
            "topics": [t.get('topic', '') for t in news.get('topics', [])],
            "url": news.get('url', '').strip(),
            "alphavantage_sentiment_score": news["overall_sentiment_score"]
        }
        for news in data["feed"]
    ]

    return news_list


def get_sentiment_from_db_by_url(conn, symbol: str, url: str):
    """
    根据 symbol + url 查询新闻情感分：
    - News.url 唯一
    - NewsPiece (news_id, ticker_id) 联合唯一
    返回：dict 或 None
    """

    with conn.cursor(pymysql.cursors.DictCursor) as cursor:

        # 1. 获取 ticker_id
        sql_ticker = """
            SELECT id
            FROM Ticker
            WHERE symbol = %s
            LIMIT 1
        """
        cursor.execute(sql_ticker, (symbol,))
        row = cursor.fetchone()

        if not row:
            print(f"[WARN] symbol={symbol} 在 Ticker 表中不存在")
            return None

        ticker_id = row["id"]

        # 2️. 根据 url 获取 news_id
        sql_news = """
            SELECT id, title, content, publish_time, source, url
            FROM News
            WHERE url = %s
            LIMIT 1
        """
        cursor.execute(sql_news, (url,))
        news_row = cursor.fetchone()

        if not news_row:
            return None

        news_id = news_row["id"]

        # 3. 根据 (news_id, ticker_id) 获取 NewsPiece
        sql_piece = """
            SELECT
                sentiment,
                sentiment_score,
                alphavantage_sentiment_score,
                sentiment_title_content,
                sentiment_score_title_content
            FROM NewsPiece
            WHERE news_id = %s
              AND ticker_id = %s
            LIMIT 1
        """
        cursor.execute(sql_piece, (news_id, ticker_id))
        piece_row = cursor.fetchone()

        if not piece_row:
            return None

    # 4. 统一返回结构（单条 dict）
    return {
        "sentiment": piece_row.get("sentiment"),
        "sentiment_score": piece_row.get("sentiment_score"),
        "sentiment_title_content": piece_row.get("sentiment_title_content"),
        "sentiment_score_title_content": piece_row.get("sentiment_score_title_content"),
        "alphavantage_sentiment_score": piece_row.get("alphavantage_sentiment_score"),
    }


def trading_decision(score):
    """根据情感分数得到交易信号"""
    if score >= 0.25:
        # print("*** 开多仓 ***")
        return 2
    elif 0.2 < score < 0.25:
        # print("*** 不动 ***")
        return 1
    elif -0.18 <= score <= 0.2:
        # print("*** 有则清仓 ***")
        return 0
    elif -0.23 < score <-0.18:
        # print("*** 不动 ***")
        return -1
    elif score<=-0.23:
        # print("*** 开空仓 ***")
        return -2


def get_yfinance_data(symbol, start, end):
    # 将ticker转成yfinance可识别的ticker，如Bitcoin -> BTC-USD
    if symbol in YFINANCE_TICKER_MAP:
        symbol = YFINANCE_TICKER_MAP[symbol]

    end = (datetime.strptime(end, "%Y-%m-%d")+timedelta(days=1)).strftime("%Y-%m-%d") # 注意yfinance是取到end前一天，所以end+1

    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False, # 显示下载进度条
        auto_adjust=True # 之前auto-adjust一直是False，改成True试一下，文档介绍：Adjust all OHLC (Open/High/Low/Close prices) automatically? (Default is True)- just leave this always as true and don’t worry about it
    )

    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.get_level_values(0)

    return df


def get_binance_data(symbol, start, timeframe):
    """美国地区限制"""
    exchange = ccxt.binance()

    ohlcv = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=exchange.parse8601(start)
    )

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df


def get_okx_data(symbol:str, start:str, timeframe:str, asset_type:str="perp"):
    """
    获取 OKX 数据
    注意：OKX 用的时间都是 UTC！！！
    :param symbol: "BTC"/"ETH"
    :param start: 起始时间，UTC（e.g.'2020-01-01T00:00:00Z'）
    :param timeframe: 时间精度，支持1m/5m/15m/30m/1h/4h/1d/1w
    :param asset_type: "spot"/"perp"/"coin_margined"
    :return: pd.Dataframe
    """
    symbol_okx = OKX_SYMBOL_MAP[asset_type][symbol] if symbol in OKX_SYMBOL_MAP[asset_type] else symbol

    exchange = ccxt.okx({'enableRateLimit': True,}) # It prevents the client from sending too many requests in a short period, avoiding bans or IP blocks.

    since = exchange.parse8601(start)
    limit = 100 # OKX 单次最大一般是 100

    all_ohlcv = []

    # 分批拉取
    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol_okx,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)

        # 下一次从最后一根 K 线之后开始
        since = ohlcv[-1][0] + 1

        # 防止死循环
        if len(ohlcv) < limit:
            break

    # 转成 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset='timestamp').set_index('timestamp')

    # UTC——>美东时间，和新闻时间对齐
    # df = df.tz_convert('US/Eastern')

    # 写入csv文件
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, f"okx_{symbol}_{start.replace(':', '-')}_{timeframe}.csv"))

    return df


def load_ohlcv_csv(path):
    """
    读取 csv 历史 OHLCV 数据
    注意：数据timestamp为UTC！！！
    """
    df = pd.read_csv(path)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    return df


def classify_sentiment(company_name, headline, strategy):
    """
    openai 新闻打分
    :param company_name:
    :param headline:
    :param strategy:
    :return:
    """
    client = openai.OpenAI(api_key="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")

    if strategy not in ["title-only", "title+content"]:
        print("策略输入错误！")
        return None

    if strategy == "title-only":
        prompt = f"""
        Forget all your previous instructions. Pretend you are a financial expert.
        You are a financial expert with stock recommendation experience.
        Answer "YES" if good news, "NO" if bad news, or "UNKNOWN" if uncertain in the first line.
        Then elaborate with one short and concise sentence on the next line.
        Is this headline good or bad for the stock price of {company_name} in the short term?

        Headline: {headline}
        """
    else:
        prompt = f"""
        Forget all your previous instructions. Pretend you are a financial expert.
        You are a financial expert with stock recommendation experience.
        Answer "YES" if good news, "NO" if bad news, or "UNKNOWN" if uncertain in the first line.
        Then elaborate with one short and concise sentence on the next line.
        Is this newspiece good or bad for the stock price of {company_name} in the short term?

        {headline}
        """

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    return content


def process_openai_report(report:str):
    """处理openai返回结果"""
    parts = report.split('\n', 1)
    label = parts[0].strip().upper()
    explanation = parts[1].strip() if len(parts) > 1 else ''

    # 映射 label ——> 分数
    if label == "YES":
        score = 1.0
    elif label == "NO":
        score = -1.0
    else:
        score = 0.0

    return explanation, score


def get_daily_signal(conn, symbol:str, cur_date:str):
    """
    根据 symbol+日期 ，获取当日交易信号
    :param conn: Connection
    :param symbol: 格式"BTC""ETH"...
    :param cur_date: str
    :return: 若新闻数量<20条，返回None，否则返回字典
    {
        "title": {
            "score": title_score,
            "signal": title_signal
        },
        "title_content": {
            "score": title_content_score,
            "signal": title_content_signal
        }
    }
    """

    # 获取AlphaVantage新闻
    news_list = get_alphavantage_news(symbol, cur_date)

    len_news_list = len(news_list)

    if len_news_list < 20:
        print(f"{cur_date}新闻数量仅{len_news_list}条，不做处理！")
        return None

    print(f"{cur_date}共获取到{len_news_list}条新闻。")

    # 从数据库获取新闻情感分
    title_scores, title_content_scores = [], []

    for news in news_list:
        result = get_sentiment_from_db_by_url(conn, symbol, news["url"])

        # 分数存在于数据库则直接用，否则 openai 打分
        # 1. Title-Only
        if not result or result["sentiment_score"] is None:
            headline = news['title']
            raw_sentiment = classify_sentiment(symbol, headline, "title-only")

            # 处理 openai 分析结果
            _, score = process_openai_report(raw_sentiment)
            title_scores.append(score)
        else:
            title_scores.append(float(result["sentiment_score"]))

        # 2. Title-Content
        if not result or result["sentiment_score_title_content"] is None:
            newspiece = f"News title: {news['title']}\nNews content: {news['content']}"
            raw_sentiment = classify_sentiment(symbol, newspiece, "title+content")

            # 处理 openai 分析结果
            _, score = process_openai_report(raw_sentiment)
            title_content_scores.append(score)
        else:
            title_content_scores.append(float(result["sentiment_score_title_content"]))

    # 计算 score、signal
    if not title_scores:
        print(f"{cur_date} 没有Title-Only新闻分数！")
        title_score = title_signal = None
    else:
        title_score = sum(title_scores) / len(title_scores)
        title_signal = trading_decision(title_score)

    if not title_content_scores:
        print(f"{cur_date} 没有Title-Content新闻分数！")
        title_content_score = title_content_signal = None
    else:
        title_content_score = sum(title_content_scores) / len(title_content_scores)
        title_content_signal = trading_decision(title_content_score)

    return {
        "title": {
            "score": title_score,
            "signal": title_signal
        },
        "title_content": {
            "score": title_content_score,
            "signal": title_content_signal
        }
    }
