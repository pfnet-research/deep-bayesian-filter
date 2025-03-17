import numpy as np
import torch
import torch.nn as nn

device = "cuda"


class Conv3DBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, imagesize
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm([out_channels, imagesize, imagesize, imagesize])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Conv3DNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.width = int(np.cbrt(input_size))
        # Initial 3D Convolutional Layer
        self.initial_conv = Conv3DBlock(
            1, 2, kernel_size=3, stride=1, padding=1, imagesize=self.width
        )

        # Intermediate Convolutional Layers with decreasing spatial dimensions
        self.conv1 = Conv3DBlock(
            2, 4, kernel_size=3, stride=2, padding=1, imagesize=self.width // 2
        )
        self.conv2 = Conv3DBlock(
            4, 8, kernel_size=3, stride=2, padding=1, imagesize=self.width // 4
        )
        self.conv3 = Conv3DBlock(
            8, 16, kernel_size=3, stride=2, padding=1, imagesize=self.width // 8
        )
        self.conv4 = Conv3DBlock(
            16, 32, kernel_size=3, stride=2, padding=1, imagesize=self.width // 16
        )

        # Final Regression Layer
        self.linear_conv1 = nn.Linear(
            4 * (self.width // 2) * (self.width // 2) * (self.width // 2), hidden_size
        )
        self.linear_conv2 = nn.Linear(
            8 * (self.width // 4) * (self.width // 4) * (self.width // 4), hidden_size
        )
        self.linear_conv3 = nn.Linear(
            16 * (self.width // 8) * (self.width // 8) * (self.width // 8), hidden_size
        )
        self.linear_conv4 = nn.Linear(
            32 * (self.width // 16) * (self.width // 16) * (self.width // 16),
            hidden_size,
        )
        self.linear2 = nn.Linear(input_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear_h1 = nn.Linear(hidden_size, hidden_size)
        self.linear_h2 = nn.Linear(hidden_size, hidden_size)
        self.linear_h3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm([input_size])

    def forward(self, x):
        original_pix = self.layernorm(x)
        x = x.reshape(len(x), 1, self.width, self.width, self.width)
        x = self.initial_conv(x)
        x = self.conv1(x)
        x_conv1 = self.linear_conv1(x.view(x.size(0), -1))
        x = self.conv2(x)
        x_conv2 = self.linear_conv2(x.view(x.size(0), -1))
        x = self.conv3(x)
        x_conv3 = self.linear_conv3(x.view(x.size(0), -1))
        x = self.conv4(x)
        x_conv4 = self.linear_conv4(x.view(x.size(0), -1))
        x = x_conv1 + x_conv2 + x_conv3 + x_conv4 + self.linear2(original_pix)
        x = self.relu(self.linear_h1(x))
        x = self.relu(self.linear_h2(x))
        x = self.relu(self.linear_h3(x))
        x = self.linear3(x)

        return x


def network_conv3D_diagonalG_init(
    z_dim, x_dim, hidden_size=10, random_seed=42, init_weight_mode=1, layer_mode=1
):
    print("network initialized as diagonal G")
    input_size = x_dim
    f_output_size = z_dim
    g_output_size = z_dim  # only diagonal parts

    torch.manual_seed(random_seed)

    f_network = Conv3DNet(input_size, hidden_size, f_output_size).to(device)
    G_network = Conv3DNet(input_size, hidden_size, g_output_size).to(device)
    # f_network.apply(init_weights)
    # G_network.apply(init_weights)
    if z_dim == 6:
        dim = G_network.linear3.weight.shape[1]
        to_add = torch.cat(
            [
                0.003 * torch.ones((1, dim)),
                0.003 * torch.ones((1, dim)),
                0.003 * torch.ones((1, dim)),
                0.01 * torch.ones((1, dim)),
                0.01 * torch.ones((1, dim)),
                0.01 * torch.ones((1, dim)),
            ]
        ).to(device)
    else:
        print("unknown z_dim")
    G_network.linear3.weight = nn.Parameter(G_network.linear3.weight + to_add)
    return f_network, G_network


def network_init_moving_MNIST_3D(
    z_dim, x_dim, momentum_for_batchnorm=0.01, hidden_size=10, random_seed=42
):
    f_network, G_network = network_conv3D_diagonalG_init(
        z_dim,
        x_dim,
        hidden_size,
    )
    return f_network, G_network
