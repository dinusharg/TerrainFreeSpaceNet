import torch
import torch.nn as nn


class PointNetRegressor(nn.Module):
    """
    PointNet-style regressor for terrain free-space prediction.

    Input:
        x: Tensor of shape [B, C, N]
           where C is typically 3 for (x, y, z)

    Output:
        Tensor of shape [B, 1] with values in [0, 1]
    """

    def __init__(self, input_dim: int = 3, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        x = self.features(x)
        x = torch.max(x, dim=2)[0]   # global max pooling -> [B, 1024]
        x = self.regressor(x)        # [B, 1]
        return x