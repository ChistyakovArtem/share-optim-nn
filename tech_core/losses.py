import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class StreamingSharpeLoss(torch.nn.Module):
    def __init__(self, asset_names, fee: float = 0.001, fees_per_share: float = 0.003, eps: float = 1e-6, intervals_per_year: int = 252*6.5*60, loss_type: str = 'sharpe'):
        super().__init__()
        self.asset_names = asset_names
        self.fee = fee
        self.fees_per_share = fees_per_share
        self.eps = eps
        self.intervals_per_year = intervals_per_year
        self.weights_sum = None
        self.loss_type = loss_type
        self.reset()

    def reset(self):
        self.pnl_log = []
        self.weights_sum = None
        self.w_prev = None

    def get_spread_estimation(self, min_prices: torch.Tensor, market_caps: torch.Tensor) -> torch.Tensor:
        """
        min_prices: torch.Tensor of shape (T, N), цены в $
        market_caps: torch.Tensor of shape (T, N), капитализации в миллиардах $
        Возвращает: оценку спреда (в $) для каждой акции во времени
        """

        # логарифмы с безопасностью
        log_price = torch.log10(min_prices.clamp(min=1e-3))
        log_cap = torch.log10(market_caps.clamp(min=1e-3))

        # Базовый спред
        base_spread = 0.01

        # Корректировки
        adj_cap = torch.clamp(0.02 - 0.005 * (log_cap - 3), min=0)
        adj_price = torch.clamp(0.01 * (1.5 - log_price), min=0)

        # Общий спред
        spread = base_spread + adj_cap + adj_price

        # Ограничим сверху до 0.1 (плохие неликвидные активы)
        return spread.clamp(max=0.1)

    def forward(self, weights: torch.Tensor, returns: torch.Tensor, prev_model_cash: torch.Tensor, min_prices: torch.Tensor = None, market_caps: torch.Tensor = None):
        """
        weights: (T, N)
        returns: (T, N)
        prev_model_cash: (T,) - количества кеша который остался от предыдущей модели
        min_prices: (T, N) - цены в $
        market_caps: (T, N) - капитализации в миллиардах $
        Возвращает: значение функции потерь
        """
        T, N = weights.shape

        returns_with_cash = torch.cat([returns, torch.zeros_like(returns[:, :1])], dim=1)  # (T, N+1)
        weights = weights * prev_model_cash

        port_ret = (weights * returns_with_cash).sum(dim=1)  # (T,)

        if self.w_prev is None:
            trans_cost = torch.zeros_like(port_ret)
        else:
            w_all = torch.cat([self.w_prev.unsqueeze(0), weights], dim=0)
            #trans_cost = torch.abs(w_all[1:, :-1] - w_all[:-1, :-1]) * (self.fees_per_share + self.get_spread_estimation(min_prices, market_caps / 1e9)) / min_prices
            trans_cost = torch.abs(w_all[1:, :-1] - w_all[:-1, :-1]) * self.fee
            trans_cost = trans_cost.sum(dim=1)

        if self.weights_sum is None:
            self.weights_sum = weights.mean(dim=0, keepdim=True)
        else:
            self.weights_sum += weights.mean(dim=0, keepdim=True)

        r_net = port_ret - trans_cost

        # Сохраняем последнее значение (detach, чтобы не раздувать граф)
        self.w_prev = weights[-1].detach()

        self.pnl_log.append(r_net.detach())  # detach здесь нормально — только для логов

        return self.compute_loss(r_net)

    def compute_loss(self, r_net: torch.Tensor):
        mean = r_net.mean()
        std = r_net.std(unbiased=False) + self.eps
        if self.loss_type == 'sharpe':
            scaler = 1
        elif self.loss_type == 'sharpe-pnl':
            scaler = mean.abs().mean() * 10000
        elif self.loss_type == 'pnl':
            scaler = std * 10000
        sharpe = mean * scaler / std * np.sqrt(self.intervals_per_year)
        return -sharpe

    def plot_whole_epoch_loss(self):
        if not self.pnl_log:
            return 0.0
        r_net = torch.cat(self.pnl_log).flatten()
        sharpe = -self.compute_loss(r_net)
        print(f"Sharpe Ratio for the epoch: {sharpe.item():.4f}")

        weights_sum = self.weights_sum / len(self.pnl_log)

        weights_sum = pd.DataFrame(weights_sum.detach().cpu().numpy().T, columns=['Weight'], index=self.asset_names).sort_values(by='Weight', ascending=False)
        print("Average Weights:\n")
        display(weights_sum)

        plt.plot(np.cumsum(r_net.cpu().numpy()), label='PnL')
        plt.title("Cumulative PnL")
        plt.xlabel("Time")
        plt.ylabel("Return")
        plt.legend()
        plt.show()
        return sharpe.item(), weights_sum

class MSEPortfolioLoss(torch.nn.Module):
    def __init__(self, fee: float = 0.001):
        super().__init__()
        self.fee = fee
        self.reset()
        self.total_mse_loss = []

    def reset(self):
        self.pnl_log = []
        self.total_mse_loss = []
        self.w_prev = None

    def forward(self, weights: torch.Tensor, returns: torch.Tensor):
        """
        weights: (T, N)
        returns: (T, N)
        """
        T, N = weights.shape

        # Добавим нулевой актив (кеш)
        returns_with_cash = torch.cat([returns, torch.zeros_like(returns[:, :1])], dim=1)  # (T, N+1)

        # Для подсчета PnL
        port_ret = (weights * returns_with_cash).sum(dim=1)  # (T,)

        if self.w_prev is None:
            trans_cost = torch.zeros_like(port_ret)
        else:
            w_all = torch.cat([self.w_prev.unsqueeze(0), weights], dim=0)
            trans_cost = self.fee * torch.abs(w_all[1:] - w_all[:-1]).sum(dim=1)

        r_net = port_ret - trans_cost
        self.w_prev = weights[-1].detach()
        self.pnl_log.append(r_net.detach())

        loss = torch.nn.functional.mse_loss(weights[:, :-1], returns_with_cash[:, :-1], reduction='mean')  # MSE loss for weights excluding cash
        self.total_mse_loss.append(loss.detach().cpu().numpy())
        return loss

    def plot_whole_epoch_loss(self):
        if not self.pnl_log:
            return 0.0
        r_net = torch.cat(self.pnl_log).flatten()
        sharpe = r_net.mean() / (r_net.std(unbiased=False) + 1e-6)

        print(f"Sharpe Ratio (MSE-guided) for the epoch: {sharpe.item():.4f}")
        print(f"Total MSE loss for the epoch: {np.mean(self.total_mse_loss):.10f}")

        plt.plot(np.cumsum(r_net.cpu().numpy()), label='PnL')
        plt.title("Cumulative PnL (MSE loss)")
        plt.xlabel("Time")
        plt.ylabel("Return")
        plt.legend()
        plt.show()
        return sharpe.item()
