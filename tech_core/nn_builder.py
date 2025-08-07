import torch
import torch.nn as nn

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

# class LSTMPortfolioAllocator(nn.Module):
#     def __init__(self, cmf_dim=50, num_assets=500, hidden_dim=32, num_layers=1):
#         super().__init__()
#         self.num_assets = num_assets

#         # LSTM по глобальным факторам
#         self.lstm = nn.LSTM(
#             input_size=cmf_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True
#         )

#         # Последний hidden state -> линейный слой
#         self.output_layer = nn.Linear(hidden_dim, num_assets + 1)

#     def forward(self, cmf, asset_features=None):
#         """
#         cmf: [batch_size, seq_len, cmf_dim]
#         asset_features: [batch_size, num_assets, asset_dim] (можно игнорировать)
#         """
#         _, (h_n, _) = self.lstm(cmf)  # h_n: [num_layers, batch_size, hidden_dim]
#         h_last = h_n[-1]  # [batch_size, hidden_dim]

#         weights = self.output_layer(h_last)  # [batch_size, num_assets + 1]
#         weights = torch.softmax(weights, dim=-1)
#         return weights

class DeepPortfolioAllocator_1(nn.Module):
    def __init__(self, cmf_dim=50, num_assets=500, asset_dim=8):
        super().__init__()
        self.num_assets = num_assets
        self.asset_dim = asset_dim

        self.cmf_net = nn.Sequential(
            nn.Linear(cmf_dim, 32),  # +1 для кеша
            nn.ELU(),
            nn.Linear(32, num_assets + 1),  # +1 для кеша
            nn.ELU()
        )

        # Эта сетка будет применяться к каждому активу одинаково
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
        asset_bias = cmf_out[:, :-1]  # [batch_size, num_assets]
        cash_bias = cmf_out[:, -1:]   # [batch_size, 1]
        asset_bias = asset_bias.unsqueeze(-1)  # [batch_size, num_assets, 1]
        concat = torch.cat([asset_features, asset_bias], dim=-1)  # [batch_size, num_assets, asset_dim + 1]
        asset_scores = self.shared_asset_net(concat).squeeze(-1)  # [batch_size, num_assets]
        all_scores = torch.cat([asset_scores, cash_bias], dim=-1)  # [batch_size, num_assets + 1]

        weights = torch.softmax(all_scores, dim=-1)

        return weights

class DeepPortfolioAllocator(nn.Module):
    def __init__(self, cmf_dim=50, asset_dim=12, num_assets=500,
                 hidden_cmf=64, hidden_asset=32, head_hidden=32):
        super().__init__()
        self.num_assets = num_assets

        # CMF encoder
        self.cmf_mlp = nn.Sequential(
            nn.Linear(cmf_dim, hidden_cmf),
            nn.ReLU(),
            nn.Linear(hidden_cmf, hidden_cmf),
            nn.ReLU()
        )
        self.cmf_mlp.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

        # Shared asset encoder
        self.asset_mlp = nn.Sequential(
            nn.Linear(asset_dim, hidden_asset),
            nn.ReLU(),
            nn.Linear(hidden_asset, hidden_asset),
            nn.ReLU()
        )
        self.asset_mlp.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

        # Head MLP (shared across assets)
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_asset + hidden_cmf, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1)  # 1 логит на актив
        )
        self.head_mlp.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

        # Cash head (Mini MLP от CMF)
        self.cash_head = nn.Sequential(
            nn.Linear(hidden_cmf, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)  # 1 логит для кеша
        )
        self.cash_head.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, cmf, asset_features):
        """
        cmf: [batch_size, cmf_dim]
        asset_features: [batch_size, num_assets, asset_dim]
        """

        batch_size = cmf.size(0)

        # CMF обработка
        cmf_embed = self.cmf_mlp(cmf)  # [batch_size, hidden_cmf]
        cmf_expand = cmf_embed.unsqueeze(1).expand(-1, self.num_assets, -1)  # [batch_size, num_assets, hidden_cmf]

        # Asset processing
        asset_embed = self.asset_mlp(asset_features)  # [batch_size, num_assets, hidden_asset]

        # Конкатенация CMF и asset embeddings
        combined = torch.cat([asset_embed, cmf_expand], dim=-1)  # [batch_size, num_assets, hidden_cmf + hidden_asset]

        # Head MLP для активов (shared)
        logits_assets = self.head_mlp(combined).squeeze(-1)  # [batch_size, num_assets]

        # Кеш логит из CMF
        cash_logit = self.cash_head(cmf_embed).squeeze(-1).unsqueeze(1)  # [batch_size, 1]

        # Итоговые логицы
        logits = torch.cat([logits_assets, cash_logit], dim=1)  # [batch_size, num_assets + 1]

        # Softmax по всем активам + кеш
        weights = torch.softmax(logits, dim=-1)  # [batch_size, num_assets + 1]

        return weights