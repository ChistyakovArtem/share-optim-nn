import torch
import torch.nn as nn

class WeightsPostprocessor(nn.Module):
    def __init__(self, max_turnover=1, max_abs_pos=10):
        super(WeightsPostprocessor, self).__init__()
        self.max_turnover = max_turnover
        self.max_abs_pos = max_abs_pos
        
    def forward(self, prev_weights, new_weights):
        weights = torch.cat([prev_weights, new_weights], dim=0)
        weights_diff = weights[1:] - weights[:-1]
        weights_diff = torch.clamp(weights_diff, min=-self.max_turnover, max=self.max_turnover)
        weights = prev_weights[-1].unsqueeze(0) + torch.cumsum(weights_diff, dim=0)
        weights = torch.clamp(weights, min=-self.max_abs_pos, max=self.max_abs_pos)
        return weights