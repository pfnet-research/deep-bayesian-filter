import torch
import torch.nn as nn

device = "cuda"


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        use_norm,
        use_skip_connections,
        activation,
        momentum_for_batchnorm,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_norm = use_norm
        self.use_skip_connections = use_skip_connections
        assert activation in ["relu", "gelu", "None"]
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if use_norm == "BatchNorm":
            self.batchnorm = nn.BatchNorm1d(
                out_features, momentum=momentum_for_batchnorm
            )
        elif use_norm == "InstanceNorm":
            self.instancenorm = nn.InstanceNorm1d(
                out_features, momentum=momentum_for_batchnorm
            )
        elif use_norm == "LayerNorm":
            self.layernorm = nn.LayerNorm([out_features])

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")

    def forward(self, x):
        x_orig = x
        x = self.linear(x)
        if self.use_norm == "BatchNorm":
            x = self.batchnorm(x)  # for BN1d

        elif self.use_norm == "InstanceNorm":
            x = self.instancenorm(x)  # for BN1d

        elif self.use_norm == "LayerNorm":
            x = self.layernorm(x)
            # x = self.batchnorm(x.unsqueeze(2)).squeeze(2)

        x = self.activation(x)
        if self.use_skip_connections and self.in_features == self.out_features:
            x = x + x_orig
        return x


class DeepNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        features,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        activation,
    ):
        super().__init__()
        assert activation in ["relu", "gelu", "None"]
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")

        self.input_size = input_size
        self.output_size = output_size
        features = list([input_size] + features + [output_size])
        self.num_blocks_linear = len(features)  # + 1
        self.use_skip_connections = use_skip_connections
        self.blocks_linear = nn.ModuleList(
            [
                LinearBlock(
                    features[i],
                    features[i + 1],
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation=activation,
                )
                for i in range(self.num_blocks_linear - 2)
            ]
            + [
                LinearBlock(
                    features[self.num_blocks_linear - 2],
                    features[self.num_blocks_linear - 1],
                    use_norm="None",
                    use_skip_connections=False,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation="None",
                )
            ]
        )

    def forward(self, x):
        for _i, l in enumerate(self.blocks_linear):
            x = l(x)
        return x


def network_init_doublependulum(
    z_dim, x_dim, momentum_for_batchnorm=0.01, hidden_size=10, random_seed=42
):
    fg_input_size = x_dim
    f_output_size = z_dim
    g_output_size = z_dim
    h_input_size = z_dim
    h_output_size = x_dim
    activation = "relu"
    use_skip_connections = True
    use_norm = "LayerNorm"
    features = [100 for i in range(10)]
    features_h = features[::-1]
    torch.manual_seed(random_seed)

    f_network = DeepNetwork(
        fg_input_size,
        f_output_size,
        features,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        activation,
    ).to(device)
    G_network = DeepNetwork(
        fg_input_size,
        g_output_size,
        features,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        activation,
    ).to(device)
    h_network = DeepNetwork(
        h_input_size,
        h_output_size,
        features_h,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        activation,
    ).to(device)
    # G_network.apply(init_weights2)
    lambdas_len = z_dim
    lambdas = torch.rand(lambdas_len) * 0.01 - 0.005  # original, 231127
    f_network_params = sum(p.numel() for p in f_network.parameters())
    G_network_params = sum(p.numel() for p in G_network.parameters())
    h_network_params = sum(p.numel() for p in h_network.parameters())
    lambdas_params = z_dim
    print(
        f"total trainable params: {f_network_params+G_network_params+h_network_params+lambdas_params}"
    )
    return f_network, G_network, h_network, lambdas
