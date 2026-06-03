import torch.nn as nn

class PointNetRegressor(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.LeakyReLU(),
            nn.Linear(16,32),
            nn.LeakyReLU(),
            nn.Linear(32,32)
        )

        self.regressor = nn.Sequential(
            nn.Linear(32,16),
            nn.LeakyReLU(),
            nn.Linear(16,3)
        )

    def forward(self,x):
        # x shape: (batch, n_points, 4)
        feat = self.point_encoder(x)        # (B,N,32)
        global_feat = feat.max(dim=1)[0]    # max pooling, kill the dimension of n_points

        out = self.regressor(global_feat)
        return out