import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        num_layers,
        activation=nn.GELU,
    ):
        super().__init__()
        assert num_layers >= 2
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_dim))
        self.layers.append(activation())

        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_dim, output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
