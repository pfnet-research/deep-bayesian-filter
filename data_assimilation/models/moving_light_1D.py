import torch
import torch.nn as nn

device = "cuda"


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def network_init(z_dim, x_dim, hidden_size=10, random_seed=42):
    input_size = x_dim
    f_output_size = z_dim
    g_output_size = z_dim

    torch.manual_seed(random_seed)

    f_network = NeuralNetwork(input_size, hidden_size, f_output_size).to(device)
    G_network = NeuralNetwork(input_size, hidden_size, g_output_size).to(device)
    # G_network.apply(init_weights2)
    return f_network, G_network


def network_init_moving_light_1D(
    z_dim, x_dim, momentum_for_batchnorm=0.01, hidden_size=10, random_seed=42
):
    f_network, G_network = network_init(
        z_dim,
        x_dim,
    )
    return f_network, G_network
