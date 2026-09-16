import torch.nn as nn

class MLP_MNIST_3(nn.Module):
    def __init__(self, p=0.3, hidden=256):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.mlp_stack(x.view(x.shape[0], -1))