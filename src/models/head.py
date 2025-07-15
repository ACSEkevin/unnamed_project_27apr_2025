import torch

from torch import nn, Tensor
from torch.nn import functional as F


class Head(nn.Module):
    """Model head"""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()

        self.cls_branch = nn.Linear(input_dim, num_classes)
        self.box_branch = MLP(input_dim, hidden_dim, 4, 3)
        self.obj_branch = nn.Linear(input_dim, 1) # durational objectness

    def forward(self, x: Tensor):
        """
        Args:
        ---
            - x: hidden state spitted by transformer, shape [N_inter, Nq, D]
        """
        _cls = self.cls_branch.forward(x)
        boxes = self.box_branch.forward(x).sigmoid()
        obj = self.obj_branch.forward(x)

        return _cls, boxes, obj


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout_p: float = 0.):
        super().__init__()
        self.num_layers = num_layers
        h = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.linear_layers = nn.ModuleList(
            nn.Linear(ins, outs)
            for ins, outs in zip(h[:-1], h[1:])
        )

        self.dropout_p = dropout_p
        self.dropouts = nn.ModuleList(
            nn.Dropout(p=dropout_p) for _ in range(num_layers)
        )

        self._init_weights()

    def _init_weights(self):
        for index, layer in enumerate(self.linear_layers):
            if isinstance(layer, nn.Linear):
                _init_weights = lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu")\
                    if index < self.num_layers - 1 else nn.init.xavier_uniform_
                _init_weights(layer.weight)
                nn.init.constant_(layer.bias, 0.05)

    def forward(self, x: Tensor):
        for i, (linear, dropout) in enumerate(zip(self.linear_layers, self.dropouts)):
            x = F.relu(linear(x)) if i < self.num_layers - 1 else linear(x)
            if self.dropout_p > 0.:
                x = dropout(x)
        return x
    

__all__ = [
    "Head", "MLP"
]