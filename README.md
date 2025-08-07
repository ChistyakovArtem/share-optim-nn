# Deep Learning on Sharpe Ratio for Mid-Frequency Trading on S&P 500 Stocks

This repository contains my implementation of the ideas presented in the article [*Deep Learning for Portfolio Optimization*](https://arxiv.org/pdf/2005.13665), which has over 200 citations, along with my adaptations, use cases, and the results of applying this technique to mid-frequency trading.

## Results

<!-- Placeholder: To be filled with bullet-point summary of results -->

## Original Article

[*Deep Learning for Portfolio Optimization*] by Zihao Zhang, Stefan Zohren, and Stephen Roberts is a compelling article demonstrating the power of neural networks for complex portfolio optimization problems.  
Traditionally, machine learning has been used to predict price movements, volatility, or autocorrelations. These predictions were then fed into stochastic models or heuristics to generate trading strategies.  
This paper, however, trains a neural network **directly** to maximize the **Sharpe Ratio** of a portfolio — accounting for transaction costs.  
Using 14 years of daily candle data and a fee level of `1e-4`, the authors developed a functioning portfolio management strategy that outperformed the market.
Their strategy achieved a Sharpe ratio of **1.96**, compared to the market Sharpe of **1.52** over the same period.

I found the idea very interesting, and since no public code implementation was available, I decided to build it myself. I also explored how this approach would work in a different setting — **mid-frequency trading** (MFT).  
I was curious to see what kinds of strategies the neural network would converge to, and under what transaction cost regimes. The results exceeded my expectations, and I’m considering turning this into a publication or using it as part of my Bachelor's thesis.

## My Case & Adaptations

I used two years of **minute-level candle data** (from July 10, 2023, to July 3, 2025) for 488 out of the 500 S&P 500 stocks.  
The stock universe was fixed as of **June 2023** to avoid lookahead bias (e.g., PLTR was excluded), and 12 stocks were missing due to delisting, mergers, or data unavailability.  
All of this data can be freely retrieved using the [Polygon.io](https://polygon.io) API.

# Hypotheses and Experiments

## I tried three different sets of fees

- **Market-estimated fees**: \$0.003 per share (like in real NYSE/NASDAQ) + spread estimation by price and market cap.  
  My strategy learned to abuse this by trading stocks with the highest absolute price (almost entirely avoiding taker fees):

  - BKNG (5,457.86)
  - FICO (1,343.12)
  - GWW (935.63)
  - AZO (4,011.25)
  - NVR (7,906.91)  
  ...

  These stocks had the highest average allocation weights.  
  I discarded this setting, since I believe that such high prices likely imply wide spreads and the effect was likely a trivial inefficiency.

> This is not the first time my model came up with a surprising idea I hadn't thought of.

- **Volume-based fee**: `1.53e-4` of USD volume (based on the best Binance futures fees).  
  This is similar to the `1e-4` fee used in the original article. All assets are treated equally.  
  All experiments were performed under this fee setup unless stated otherwise.

- **Scaled-up fees**: `1.53e-4 * k`  
  What happens if fees are so high that directional (alpha-style) trading becomes impossible?  
  Will the model learn to do actual **portfolio management** (beta-style allocation)?

## Models

### Simple Allocator

```python
class SimplePortfolioAllocator(nn.Module):
    def __init__(self, cmf_dim=50, num_assets=500):
        super().__init__()
        self.num_assets = num_assets
        self.tmp_simple_linear = nn.Linear(cmf_dim, num_assets + 1)

    def forward(self, cmf, asset_features):
        """
        cmf: [batch_size, cmf_dim]
        asset_features: [batch_size, num_assets, asset_dim]
        """
        weights = self.tmp_simple_linear(cmf)
        weights = torch.softmax(weights, dim=-1)
        return weights
```

### Deep Allocator

```python
class DeepPortfolioAllocator_1(nn.Module):
    def __init__(self, cmf_dim=50, num_assets=500, asset_dim=8):
        super().__init__()
        self.num_assets = num_assets
        self.asset_dim = asset_dim

        self.cmf_net = nn.Sequential(
            nn.Linear(cmf_dim, 32),
            nn.ELU(),
            nn.Linear(32, num_assets + 1),
            nn.ELU()
        )

        self.shared_asset_net = nn.Sequential(
            nn.Linear(asset_dim + 1, 4),
            nn.ELU(),
            nn.Linear(4, 1)
        )

    def forward(self, cmf, asset_features):
        """
        cmf: [batch_size, cmf_dim]
        asset_features: [batch_size, num_assets, asset_dim]
        """
        batch_size = cmf.shape[0]
        cmf_out = self.cmf_net(cmf)  # [batch_size, num_assets + 1]
        asset_bias = cmf_out[:, :-1].unsqueeze(-1)  # [batch_size, num_assets, 1]
        cash_bias = cmf_out[:, -1:]  # [batch_size, 1]

        concat = torch.cat([asset_features, asset_bias], dim=-1)  # [batch_size, num_assets, asset_dim + 1]
        asset_scores = self.shared_asset_net(concat).squeeze(-1)  # [batch_size, num_assets]
        all_scores = torch.cat([asset_scores, cash_bias], dim=-1)  # [batch_size, num_assets + 1]

        weights = torch.softmax(all_scores, dim=-1)
        return weights
```

# Pipelines & Tech Core

In order to fit the neural network on this large amount of data with a non-trivial loss function, I had to perform data loading, time-series feature construction, and loss evaluation on smaller chunks.

- The batch size was set to **1000** (minute candles). Increasing this number did not improve performance but slowed down training.  
  A model trading on minute frequency does not need to see the market state over two full days.

- I implemented **advanced CSV mapping with offsets** to retrieve a sample of size `k=1000` from `O(k)` time, rather than `O(n)` where `n ≈ 200,000` (the total number of rows).  
  The Sharpe ratio loss was also computed in chunks to avoid storing large gradients and overflowing RAM.
```python
import os
import io
import pandas as pd

class FastCSVChunkReader:
    def __init__(self, path, padding=0, batch_size=1000, has_header=True):
        self.path = path
        self.padding = padding
        self.batch_size = batch_size
        self.offsets = []
        self.has_header = has_header
        self._build_offsets()
        self.current_idx = 0
        self.asset_names = []
        self.end_idx = len(self.offsets) - 1

    def _build_offsets(self):
        with open(self.path, 'rb') as f:
            pos = f.tell()
            if self.has_header:
                f.readline()  # skip header
                pos = f.tell()
            self.offsets.append(pos)
            while line := f.readline():
                self.offsets.append(f.tell())

        self.total_rows = len(self.offsets)

    def __iter__(self):
        return self

    def set_split(self, start_row: int, end_row: int):
        self.current_idx = start_row
        self.end_idx = end_row

    def __next__(self):

        if self.current_idx >= self.end_idx:
            raise StopIteration

        start = self.current_idx
        end = min(self.current_idx + self.batch_size + self.padding, self.end_idx)

        with open(self.path, 'rb') as f:
            f.seek(self.offsets[start])
            lines = f.read(self.offsets[end] - self.offsets[start]).decode('utf-8')
            if self.has_header:
                with open(self.path, 'r') as h:
                    header = h.readline()
                result = header + lines
            else:
                result = lines

        self.current_idx += self.batch_size
        tmp = pd.read_csv(io.StringIO(result)).set_index('Unnamed: 0').rename_axis(None)
        tmp.index = pd.to_datetime(tmp.index)
        self.asset_names = tmp.columns.tolist()
        return tmp

    def reset(self):
        self.current_idx = 0
```

- Additionally, **padding** was added to enable time-series feature computation based on previous observations, ensuring **stationarity** of features across different positions in the batch.

# Features

## Common Market Features

I used **market-cap-weighted returns** across 18 different subsets of assets. Below is the code used to construct common market features:

```python
market_caps = self.min_prices.values * self.info['Shares'].values
market_shares = market_caps / market_caps.sum(axis=1, keepdims=True)
market_shares = pd.DataFrame(market_shares, index=self.min_prices.index, columns=self.min_prices.columns)

returns = self.min_prices.pct_change().fillna(0)
returns[self.min_prices.index.to_series().dt.date.shift(1) != self.min_prices.index.to_series().dt.date] = 0

# Info for loss
self.future_returns = self.min_prices.shift(-1).pct_change().fillna(0)
self.future_returns[self.min_prices.index.to_series().shift(-1).dt.date != self.min_prices.index.to_series().dt.date] = 0
self.minute_prices = self.min_prices
self.market_caps = pd.DataFrame(market_caps, index=self.min_prices.index, columns=self.min_prices.columns)
```

Then I defined masks over different industries, sectors and other subsets.
The classification of sectors and industries follows the [GICS](https://www.msci.com/our-solutions/indexes/gics) (Global Industry Classification Standard), used by MSCI and S&P.


```python
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
```

I also added rolling averages and standard deviations of these weighted returns:

```python
for horizon in [5, 10, 30]:
    weighted_returns_tmp_mean = weighted_returns_orig.rolling(window=horizon).mean().add_suffix(f'_mean_{horizon}')
    weighted_returns_tmp_std = weighted_returns_orig.rolling(window=horizon).std().add_suffix(f'_std_{horizon}')
```
## Asset-Specific Features

Some architectures use **asset-specific features** to predict the future performance of each asset — and therefore to adjust the weight allocation on a per-asset basis.

These features are calculated from historical price movements of each asset individually:

```python
def get_asset_specific_features(self):
    features_list = []

    pct_change = self.min_prices.pct_change().fillna(0)
    pct_change[self.min_prices.index.to_series().shift(1) != self.min_prices.index.to_series()] = 0

    features_list.append(pct_change)                           # Instant returns
    features_list.append(pct_change.rolling(window=10).std())  # 10-minute volatility
    features_list.append(pct_change.rolling(window=20).std())  # 20-minute volatility
    features_list.append(pct_change.rolling(window=10).mean()) # 10-minute trend
    features_list.append(pct_change.rolling(window=20).mean()) # 20-minute trend

    return np.stack([f.values for f in features_list], axis=2)  # shape: (n_time, n_assets, n_features)
```

These features are used as the `asset_features` input in architectures like `DeepPortfolioAllocator`, where each asset is processed with its own features and combined with global context.

# Future Ideas to work with

- **Change of constraints**  
  As we see in results, the best model just found one asset and abused it. Since we trade MFT (and not a billion-dollar PM strategy), we should move to assigning each instrument the max position and max delta position (to account for liquidity).  
  These constraints can be written as:

  ```
  Sum(abs(w))       <= C1  
  Max(abs(w))       <= C2  
  Sum(abs(delta w)) <= C3  
  Max(abs(delta w)) <= C4
  ```

- **Stacking of Strategies**  
  If we try to optimize the Sharpe Ratio, our strategy tends to stay in position in a small subset of assets.  
  This way, by building a directional strategy, we remove the possibility for a PM strategy alongside it.  
  This can be fixed by trying to make a new strategy with new constraints such that **together** they satisfy the constraints above.  
  This can be achieved by:
  - Iteratively adding strategies
  - Or by fitting weights for multiple strategies together

  I tried the first option, but due to a cash restriction (I forced it to close a position if the first strategy wanted to open it), the model downgraded to holding cash.

- **Convergence and model size**  
  Right now, with ~100 common market features and 488 assets, the simplest model has `100 × 488` parameters.  
  We can try to reduce complexity by transforming the 100 features into a market embedding and stacking it alongside asset-specific features.  
  However, all these architectures fail to converge.  
  It may be fixed by first training the model on MSE for future returns and then fine-tuning it with Sharpe Ratio loss — but this is a task for upcoming months.

- **Using CatBoost predictors as features**  
  Since on Kaggle, CatBoost and similar GBMs are often superior to Neural Networks, they can be used for the **forecasting** part,  
  with a Neural Network doing the **allocation** part (instead of traditionally used heuristics).

- **Encoding asset structure**  
  Right now, the model selects a small portion of assets with no visible grouping (besides the moment where, with "market fees", the model selected those with the highest absolute price).  
  Probably, the inefficiencies in these assets do not depend on any conventional market features (like grouping by capitalization, sector, etc.).  
  This limits architectures — I have to assume that the model must fit different weights for different assets at some point, with probably no way to fix it just by adding features.
