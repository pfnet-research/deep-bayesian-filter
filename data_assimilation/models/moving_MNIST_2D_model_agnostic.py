import numpy as np
import torch
import torch.nn as nn

device = "cuda"


class Conv2D_deep(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4 * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.width = int(np.sqrt(input_size))
        self.layernorm1 = nn.LayerNorm([4, self.width, self.width])
        self.layernorm2 = nn.LayerNorm([hidden_size])

    def forward(self, x):
        x = x.reshape(len(x), self.width, self.width)
        original_pixels = x.unsqueeze(1)
        x = self.layernorm1(torch.relu(self.conv1(original_pixels)))
        x = self.layernorm1(torch.relu(self.conv2(x))) + x
        x = self.layernorm1(torch.relu(self.conv3(x))) + x
        x = self.layernorm1(torch.relu(self.conv4(x))) + x
        x = self.layernorm1(torch.relu(self.conv5(x))) + x
        x = self.layernorm1(torch.relu(self.conv6(x))) + x
        x = self.layernorm1(torch.relu(self.conv7(x))) + x
        x = self.layernorm1(torch.relu(self.conv8(x))) + x
        x = self.layernorm1(torch.relu(self.conv9(x))) + x
        x = self.layernorm1(torch.relu(self.conv10(x))) + x
        x = x.view(x.size(0), -1)
        x = self.layernorm2(torch.relu(self.fc1(x)))
        x = self.layernorm2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class Conv2D_deep_inv(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        print("network: NeuralNetwork_conv_light")
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.width = int(np.sqrt(output_size))
        self.layernorm1 = nn.LayerNorm([hidden_size])
        self.layernorm2 = nn.LayerNorm([output_size])
        self.layernorm3 = nn.LayerNorm([4, self.width, self.width])

    def forward(self, x):
        x = self.layernorm1(torch.relu(self.fc1(x)))
        x = self.layernorm1(torch.relu(self.fc2(x)))
        x = self.layernorm2(torch.relu(self.fc3(x)))
        x = x.reshape(len(x), 1, self.width, self.width)
        x = self.layernorm3(torch.relu(self.conv1(x)))
        x = self.layernorm3(torch.relu(self.conv2(x))) + x
        x = self.layernorm3(torch.relu(self.conv3(x))) + x
        x = self.layernorm3(torch.relu(self.conv4(x))) + x
        x = self.layernorm3(torch.relu(self.conv5(x))) + x
        x = self.layernorm3(torch.relu(self.conv6(x))) + x
        x = self.layernorm3(torch.relu(self.conv7(x))) + x
        x = self.layernorm3(torch.relu(self.conv8(x))) + x
        x = self.layernorm3(torch.relu(self.conv9(x))) + x
        x = torch.relu(self.conv10(x))
        x = x.view(x.size(0), -1)
        return x


def network_init_fhKG_Conv2d(z_dim, x_dim, random_seed):
    fg_input_size = x_dim
    f_output_size = z_dim
    g_output_size = z_dim
    h_input_size = z_dim
    h_output_size = x_dim
    torch.manual_seed(random_seed)
    f_network = Conv2D_deep(fg_input_size, 50, f_output_size).to(device)
    G_network = Conv2D_deep(fg_input_size, 50, g_output_size).to(device)
    h_network = Conv2D_deep_inv(h_input_size, 50, h_output_size).to(device)
    lambdas_len = z_dim
    lambdas = torch.rand(lambdas_len) * 0.01 - 0.005
    return f_network, G_network, h_network, lambdas


def network_init_moving_MNIST_2D_model_agnostic(z_dim, x_dim, random_seed=42):
    f_network, G_network, h_network, lambdas = network_init_fhKG_Conv2d(
        z_dim,
        x_dim,
        random_seed=random_seed,
    )
    return f_network, G_network, h_network, lambdas
