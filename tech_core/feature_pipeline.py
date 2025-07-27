import pandas as pd
import numpy as np
import polars as pl
from tech_core.reader import FastCSVChunkReader
import numpy as np
import pandas as pd

class FeaturesPipeline:
    def __init__(self, path_to_data: str, padding: int = 20, chunk_size: int = 1000):
        self.path_to_min_prices = path_to_data + 'minute_prices.csv'
        self.path_to_companies_info = path_to_data + 'companies_info.csv'

        self.info = pd.read_csv(self.path_to_companies_info, index_col=0)
        self.reader = FastCSVChunkReader(self.path_to_min_prices, padding=padding, chunk_size=chunk_size)
        self.padding = padding
        self.chunk_size = chunk_size

    def __iter__(self):
        return self

    def __next__(self):
        min_prices = next(self.reader)  # это уже DataFrame
        self.min_prices = min_prices  # чтобы использовать в методах ниже

        common_feats = self.get_common_market_features()
        asset_feats = self.get_asset_specific_features()

        # отсекаем padding
        common_feats = common_feats.iloc[-self.chunk_size:]
        asset_feats = asset_feats[-self.chunk_size:, :, :]  # (total, m, f) -> (chunk_size, m, f)
        self.future_returns = self.future_returns.iloc[-self.chunk_size:]

        return common_feats, asset_feats, self.future_returns

    def reset(self):
        self.reader.reset()

    def get_common_market_features(self):
        market_caps = self.min_prices.values * self.info['Shares'].values
        market_shares = market_caps / market_caps.sum(axis=1, keepdims=True)
        market_shares = pd.DataFrame(
            market_shares,
            index=self.min_prices.index,
            columns=self.min_prices.columns
        )

        returns = self.min_prices.pct_change().fillna(0)
        returns[self.min_prices.index.to_series().dt.date.shift(1) != self.min_prices.index.to_series().dt.date] = 0

        self.future_returns = self.min_prices.shift(-1).pct_change().fillna(0)
        self.future_returns[self.min_prices.index.to_series().shift(-1).dt.date != self.min_prices.index.to_series().dt.date] = 0

        bool_masks = {
            'Ind:Semiconductors': self.info['Industry'] == 'Semiconductors',
            'Ind:Semiconductors without NVDA': (self.info['Industry'] == 'Semiconductors') & (self.info.index != 'NVDA'),
            'Ind:Integrated Oil & Gas': self.info['Industry'] == 'Integrated Oil & Gas',
            'Ind:Diversified Banks': self.info['Industry'] == 'Diversified Banks',
            'Ind:Investment Banking & Brokerage': self.info['Industry'] == 'Investment Banking & Brokerage',
            'Ind:Aerospace & Defense': self.info['Industry'] == 'Aerospace & Defense',
            'Ind:Financial Exchanges & Data': self.info['Industry'] == 'Financial Exchanges & Data',
            'Ind:Pharmaceuticals': self.info['Industry'] == 'Pharmaceuticals',

            'Sec:Energy': self.info['Sector'] == 'Energy',
            'Sec:Industrials': self.info['Sector'] == 'Industrials',
            'Sec:Consumer Discretionary': self.info['Sector'] == 'Consumer Discretionary',
            'Sec:Health Care': self.info['Sector'] == 'Health Care',
            'Sec:Financials': self.info['Sector'] == 'Financials',
            'Sec:Information Technology': self.info['Sector'] == 'Information Technology',
            'Sec:Communication Services': self.info['Sector'] == 'Communication Services',
            'Sec:Real Estate': self.info['Sector'] == 'Real Estate',

            'Add:All': pd.Series(True, index=self.info.index),
            'Add:Mag7': self.info.index.isin(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']),
        }

        bool_masks = pd.DataFrame(bool_masks, index=self.info.index)
        weighted_returns = (
            (returns.values * market_shares.values) @ bool_masks.values
        ) / (
            market_shares.values @ bool_masks.values
        )
        return pd.DataFrame(
            weighted_returns,
            index=self.min_prices.index,
            columns=bool_masks.columns
        )

    def get_asset_specific_features(self):
        features_list = []

        pct_change = self.min_prices.pct_change().fillna(0)
        pct_change[self.min_prices.index.to_series().shift(1) != self.min_prices.index.to_series()] = 0

        features_list.append(pct_change)
        features_list.append(pct_change.rolling(window=10).std())
        features_list.append(pct_change.rolling(window=20).std())
        features_list.append(pct_change.rolling(window=10).mean())
        features_list.append(pct_change.rolling(window=20).mean())

        return np.stack([f.values for f in features_list], axis=2)  # shape: (n, m, f)