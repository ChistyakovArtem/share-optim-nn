import pandas as pd
import numpy as np
import polars as pl
from tech_core.reader import FastCSVChunkReader
from tech_core.split_manager import SplitManager

class FeaturesPipeline:
    def __init__(self, path_to_data: str, padding: int = 20, chunk_size: int = 1000,
                 split_dates=None, split_names=None):
        self.path_to_min_prices = path_to_data + 'minute_prices.csv'
        self.path_to_companies_info = path_to_data + 'companies_info.csv'

        self.info = pd.read_csv(self.path_to_companies_info, index_col=0)
        self.reader = FastCSVChunkReader(self.path_to_min_prices, padding=padding, chunk_size=chunk_size)
        self.padding = padding
        self.chunk_size = chunk_size

        self.split_manager = None
        self.splits = None
        if split_dates and split_names:
            self.split_manager = SplitManager(split_dates, split_names)
            self._build_splits()

    def _build_splits(self):
        # Считываем все индексы по времени
        all_times = []
        self.reader.reset()
        while True:
            try:
                chunk = next(self.reader)
                all_times.extend(chunk.tail(len(chunk) - self.padding).index.tolist())
            except StopIteration:
                break

        self.all_times = all_times
        self.splits = {
            name: self.split_manager.get_split_bounds(all_times, name)
            for name in self.split_manager.split_names
        }

    def iterate(self, split_name):
        if self.splits is None or split_name not in self.splits:
            raise ValueError(f"Unknown split {split_name}")
        start, end = self.splits[split_name]
        self.reader.reset()
        
        self.reader.set_split(start, end)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        min_prices = next(self.reader)
        self.min_prices = min_prices

        common_feats = self.get_common_market_features()
        asset_feats = self.get_asset_specific_features()

        # отсекаем паддинг
        common_feats = common_feats.iloc[-self.chunk_size:]
        asset_feats = asset_feats[-self.chunk_size:, :, :]
        self.future_returns = self.future_returns.iloc[-self.chunk_size:]

        return common_feats, asset_feats, self.future_returns, self.min_prices[-self.chunk_size:], self.market_caps[-self.chunk_size:]

    def reset(self):
        self.reader.reset()

    def get_common_market_features(self):
        market_caps = self.min_prices.values * self.info['Shares'].values
        market_shares = market_caps / market_caps.sum(axis=1, keepdims=True)
        market_shares = pd.DataFrame(market_shares, index=self.min_prices.index, columns=self.min_prices.columns)

        returns = self.min_prices.pct_change().fillna(0)
        returns[self.min_prices.index.to_series().dt.date.shift(1) != self.min_prices.index.to_series().dt.date] = 0

        # info for loss
        self.future_returns = self.min_prices.shift(-1).pct_change().fillna(0)
        self.future_returns[self.min_prices.index.to_series().shift(-1).dt.date != self.min_prices.index.to_series().dt.date] = 0
        self.minute_prices = self.min_prices
        self.market_caps = market_caps

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

        weighted_returns_orig = pd.DataFrame(weighted_returns, index=self.min_prices.index, columns=bool_masks.columns)
        weighted_returns = weighted_returns_orig.copy()
        for horizon in [5, 10, 30]:
            weighted_returns_tmp_mean = weighted_returns_orig.rolling(window=horizon).mean().add_suffix(f'_mean_{horizon}')
            weighted_returns_tmp_std = weighted_returns_orig.rolling(window=horizon).std().add_suffix(f'_std_{horizon}')
            weighted_returns = pd.concat([weighted_returns, weighted_returns_tmp_mean, weighted_returns_tmp_std], axis=1)
        
        return weighted_returns


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