import torch
import torch.nn.functional as F


class Network(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1000, output_dim=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc3(x))
        x = self.fc2(x)
        return x
