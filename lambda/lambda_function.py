"""
AAPL日次データ更新 Lambda
historical_data.ipynb のデータ取得+特徴量エンジニアリングを移植。
毎営業日実行し、最新の feature_dataset.csv を S3 に上書き保存する。

必要なレイヤー:
  1. AWSSDKPandas-Python312 (AWS管理レイヤー: pandas/numpy)
  2. yfinance カスタムレイヤー
環境変数:
  BUCKET (省略時: apple-data-2026-sagemaker)
"""
import os

# yfinanceがキャッシュを書こうとするため、書き込み可能な/tmpをHOMEにする(Lambda対策)
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import io
import boto3
import numpy as np
import pandas as pd
import yfinance as yf

BUCKET = os.environ.get("BUCKET", "apple-data-2026-sagemaker")
KEY = "feature_dataset.csv"
TICKER = "AAPL"
OUTLIER_CUTOFF = 0.01
LAGS = [1, 5, 21, 63, 252]
VOL_WINDOWS = [5, 21, 63, 252]


def volatility(returns, n=252):
    return np.std(returns, ddof=1) * np.sqrt(n)


def build_features(prices: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame(index=prices.index)

    # 1. 外れ値クリップ + 幾何平均で日次相当に正規化したリターン
    for lag in LAGS:
        returns = prices.pct_change(lag)
        returns = returns.clip(
            lower=returns.quantile(OUTLIER_CUTOFF),
            upper=returns.quantile(1 - OUTLIER_CUTOFF),
        )
        data[f"return_{lag}d"] = (1 + returns) ** (1 / lag) - 1
    data = data.dropna()

    # 2. デイリーリターンのラグ
    for t in range(1, 5):
        data[f"return_1d_t-{t}"] = data.return_1d.shift(t)

    # 3. モメンタム (Stefan Jansen方式)
    for lag in [5, 21, 63, 252]:
        data[f"momentum_{lag}d"] = data[f"return_{lag}d"] - data["return_1d"]
    data["momentum_5_21d"] = data["momentum_21d"].sub(data["return_5d"], axis=0)
    data["momentum_5_63d"] = data["momentum_63d"].sub(data["return_5d"], axis=0)
    data["momentum_21_252d"] = data["momentum_252d"].sub(data["return_21d"], axis=0)

    # 4. カレンダー特徴量 (既存CSVの列名 'presidetal_cycle' に合わせる)
    data["presidetal_cycle"] = (data.index.year % 4) == 0

    # 5. ボラティリティ (年率化)
    daily_returns = prices.pct_change()
    for window in VOL_WINDOWS:
        data[f"volatility_{window}d"] = (
            daily_returns.rolling(window=window).apply(volatility, raw=True)
        )

    return data.dropna()


def lambda_handler(event, context):
    # データ取得 (直近5年)
    raw = yf.download(TICKER, period="5y", interval="1d", auto_adjust=True)
    prices = raw["Close"]
    if isinstance(prices, pd.DataFrame):  # MultiIndex列対策
        prices = prices.squeeze("columns")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"

    features = build_features(prices)

    # S3へ保存
    buf = io.StringIO()
    features.to_csv(buf, index_label="Date")
    boto3.client("s3").put_object(Bucket=BUCKET, Key=KEY, Body=buf.getvalue())

    msg = (
        f"OK: {len(features)} rows, {features.shape[1]} cols, "
        f"last date {features.index.max().date()} -> s3://{BUCKET}/{KEY}"
    )
    print(msg)
    return {"statusCode": 200, "body": msg}
