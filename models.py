import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_layers, use_dropout=False, dropout_p=0.3):
        """
        hidden_layers: e.g. [256, 128] creates 784→256→128→10
        """
        super().__init__()
        layers = []
        in_features = 784
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_p))
            in_features = out_features
        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28×28 → 784
        return self.network(x)


MODEL_CONFIGS = {
    "A_SGD_Kucuk": {
        "hidden_layers": [256, 128],
        "use_dropout": False,
        "optimizer": "SGD",
        "lr": 0.01,
        "label": "Model A — SGD, 2 Katman (256→128)",
    },
    "B_Adam_Kucuk": {
        "hidden_layers": [256, 128],
        "use_dropout": False,
        "optimizer": "Adam",
        "lr": 0.001,
        "label": "Model B — Adam, 2 Katman (256→128)",
    },
    "C_Adam_Derin": {
        "hidden_layers": [512, 256, 128],
        "use_dropout": False,
        "optimizer": "Adam",
        "lr": 0.001,
        "label": "Model C — Adam, 3 Katman (512→256→128)",
    },
    "D_Adam_Dropout": {
        "hidden_layers": [512, 256, 128],
        "use_dropout": True,
        "optimizer": "Adam",
        "lr": 0.001,
        "label": "Model D — Adam, 3 Katman + Dropout",
    },
}
