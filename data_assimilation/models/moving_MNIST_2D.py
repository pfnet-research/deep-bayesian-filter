import numpy as np
import torch
import torch.nn as nn

device = "cuda"


class NeuralNetwork_conv_light(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        print("network: NeuralNetwork_conv_light")
        super(NeuralNetwork_conv_light, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)  # 64*64->32*32
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)  # 32*32->16*16
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)  # 16*16->8*8
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 8*8->4*4
        total_dim = (
            input_size
            + (input_size * 2 / 4)
            + (input_size * 4 / 16)
            + (input_size * 8 / 64)
            + (input_size * 16 / 256)
        )
        self.fc1 = nn.Linear(int(total_dim), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.width = int(np.sqrt(input_size))

    def forward(self, x):
        x = x.reshape(len(x), self.width, self.width)
        original_pixels = x.unsqueeze(1)
        conv1_output = torch.relu(self.conv1(original_pixels))
        conv2_output = torch.relu(self.conv2(conv1_output))
        conv3_output = torch.relu(self.conv3(conv2_output))
        conv4_output = torch.relu(self.conv4(conv3_output))
        flattened_original_pixels = original_pixels.view(original_pixels.size(0), -1)
        flattened_conv1_output = conv1_output.view(conv1_output.size(0), -1)
        flattened_conv2_output = conv2_output.view(conv2_output.size(0), -1)
        flattened_conv3_output = conv3_output.view(conv3_output.size(0), -1)
        flattened_conv4_output = conv4_output.view(conv4_output.size(0), -1)
        combined_input = torch.cat(
            (
                flattened_original_pixels,
                flattened_conv1_output,
                flattened_conv2_output,
                flattened_conv3_output,
                flattened_conv4_output,
            ),
            dim=1,
        )
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def network_conv_diagonalG_init(
    z_dim, x_dim, hidden_size=10, random_seed=42, init_weight_mode=1, layer_mode=1
):
    print("network initialized as diagonal G")
    input_size = x_dim
    f_output_size = z_dim
    g_output_size = z_dim  # only diagonal parts
    f_network = NeuralNetwork_conv_light(input_size, hidden_size, f_output_size).to(
        device
    )
    G_network = NeuralNetwork_conv_light(input_size, hidden_size, g_output_size).to(
        device
    )
    return f_network, G_network


def network_init_moving_MNIST_2D(
    z_dim, x_dim, momentum_for_batchnorm=0.01, hidden_size=10, random_seed=42
):
    f_network, G_network = network_conv_diagonalG_init(
        z_dim,
        x_dim,
    )
    return f_network, G_network
