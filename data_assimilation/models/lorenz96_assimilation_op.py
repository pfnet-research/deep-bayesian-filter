import copy
import math
from typing import List  # NOQA

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# import time
from torch import profiler  # NOQA
from torch import einsum, nn

torch.set_default_dtype(torch.float32)
#device = "cuda"


class MyConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode="circular"):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv_t = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, padding=0
        )
        assert padding_mode in ["circular", "constant", "replicate", "reflect"]
        self.padding_mode = padding_mode
        self.weight = self.conv_t.weight
        self.bias = self.conv_t.bias

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding), mode=self.padding_mode)
        x = self.conv_t(x)
        if self.padding > 0:
            x = x[:, :, 2 * self.padding : -2 * self.padding]
        else:
            x = x
        return x


# experimental
class ViT_encoder_for_Lorenz96(nn.Module):
    def __init__(
        self,
        x_dim,
        patch_size,
        D,
        hidden_dim,
        nhead,
        num_layers,
        dim_feedforward,
        activation,
        norm_first,
        input_normalize_factor=1.0,
    ):
        # print(
        #     f"ViT_encoder_for_Lorenz96, x_dim={x_dim}, patch_size={patch_size}, "
        #     f"D={D}, hidden_dim={hidden_dim}, nhead={nhead}, num_layers={num_layers}, "
        #     f"dim_feedforward={dim_feedforward}, activation={activation}, "
        #     f"norm_first={norm_first}, input_normalize_factor={input_normalize_factor}"
        # )
        super().__init__()
        assert x_dim % patch_size == 0, "x_dim needs be divisible by patch_size"
        self.x_dim = x_dim
        self.patch_size = patch_size
        self.N = self.x_dim // self.patch_size
        self.D = D
        self.E = torch.rand(self.N, self.D, requires_grad=True, device="cuda")
        self.position = torch.arange(self.N, device="cuda").unsqueeze(1)
        self.div_term = torch.exp(
            torch.arange(0, self.D, 2, device="cuda") * (-math.log(10000.0) / self.D)
        )
        self.mul = torch.arange(self.D, device="cuda")
        self.E_pos = torch.zeros(self.N, self.D, device="cuda")
        self.E_pos[:, 0::2] = torch.sin(self.position * self.div_term)
        self.E_pos[:, 1::2] = torch.cos(self.position * self.div_term)
        # self.E_pos = self.mul*torch.sin(self.position+1)
        # print("self.E_pos: ", self.E_pos)
        # print("self.E_pos.shape: ", self.E_pos.shape)
        assert activation in ["relu", "gelu", "None"]
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(self.N * self.D, hidden_dim)
        self.input_normalize_factor = input_normalize_factor

    def tokenize_and_embed(self, x):
        xp = x.unfold(1, self.patch_size, self.patch_size)
        embedded = einsum("nd,bnp->bnd", self.E, xp)
        return embedded + self.E_pos

    def forward(self, x):
        x = x * self.input_normalize_factor
        embedded_with_positional_encoding = self.tokenize_and_embed(x)
        transformer_output = self.transformer_encoder(embedded_with_positional_encoding)
        flattened = rearrange(transformer_output, "b c w -> b (c w)")
        output = self.fc(flattened)
        return output


# experimental
class ViT_encoder_for_Lorenz96_TimeSeries(nn.Module):
    def __init__(
        self,
        x_dim,
        patch_size,
        time_series_length,
        D,
        hidden_dim,
        nhead,
        num_layers,
        dim_feedforward,
        activation,
        norm_first,
        input_normalize_factor,
    ):
        # print(
        #     f"ViT_encoder_for_Lorenz96_TimeSeries, x_dim={x_dim}, "
        #     f"patch_size={patch_size}, time_series_length={time_series_length}, "
        #     f"D={D}, hidden_dim={hidden_dim}, nhead={nhead}, "
        #     f"num_layers={num_layers}, dim_feedforward={dim_feedforward}, "
        #     f"activation={activation}, norm_first={norm_first}, "
        #     f"input_normalize_factor={input_normalize_factor}"
        # )
        assert x_dim % patch_size == 0, "x_dim needs be divisible by patch_size"
        super().__init__()
        # take all x_dim as one input and
        self.x_dim = x_dim
        self.patch_size = patch_size
        self.N_spatial = self.x_dim // self.patch_size
        self.time_series_length = time_series_length
        self.D = D
        self.E = torch.rand(
            self.time_series_length,
            self.N_spatial,
            self.D,
            requires_grad=True,
            device="cuda",
        )  # separate dimensions for time_series_length and spatial position
        self.position = torch.arange(self.N_spatial, device="cuda").unsqueeze(1)
        self.time_sequence = torch.arange(time_series_length, device="cuda").unsqueeze(
            1
        )
        self.div_term = torch.exp(
            torch.arange(0, self.D // 2, 2, device="cuda")
            * (-math.log(10000.0) / self.D)
        )
        self.E_pos = torch.zeros(
            self.time_series_length, self.N_spatial, self.D, device="cuda"
        )
        for _i in range(self.time_series_length):
            self.E_pos[:, :, 0 : self.D // 2 : 2] = (
                torch.sin(self.time_sequence * self.div_term)
                .unsqueeze(1)
                .repeat(1, self.N_spatial, 1)
            )
            self.E_pos[:, :, 1 : self.D // 2 : 2] = (
                torch.cos(self.time_sequence * self.div_term)
                .unsqueeze(1)
                .repeat(1, self.N_spatial, 1)
            )
            self.E_pos[:, :, self.D // 2 :: 2] = (
                torch.sin(self.position * self.div_term)
                .unsqueeze(0)
                .repeat(self.time_series_length, 1, 1)
            )
            self.E_pos[:, :, self.D // 2 + 1 :: 2] = (
                torch.cos(self.position * self.div_term)
                .unsqueeze(0)
                .repeat(self.time_series_length, 1, 1)
            )
        # print("self.E_pos: ", self.E_pos)
        # print("self.E_pos.shape: ", self.E_pos.shape)
        assert activation in ["relu", "gelu", "None"]
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(
            self.N_spatial * self.time_series_length * self.D, hidden_dim
        )
        self.input_normalize_factor = input_normalize_factor

    def tokenize_and_embed(self, x):
        xp = x.unfold(2, self.patch_size, self.patch_size).reshape(
            len(x), self.time_series_length, self.N_spatial, self.patch_size
        )
        embedded = einsum("tnd,btnp->btnd", self.E, xp)
        embedded_with_position = embedded + self.E_pos
        return embedded_with_position.reshape(
            len(x), self.time_series_length * self.N_spatial, self.D
        )  # reshape here

    def forward(self, x):
        x = x * self.input_normalize_factor
        embedded_with_positional_encoding = self.tokenize_and_embed(x)
        transformer_output = self.transformer_encoder(embedded_with_positional_encoding)
        flattened = rearrange(transformer_output, "b c w -> b (c w)")
        output = self.fc(flattened)
        assert not torch.isnan(output).any().item()
        return output


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        padding_mode,
        use_norm,
        use_skip_connections,
        activation,
        momentum_for_batchnorm,
        z_dim_for_layernorm=40,
    ):
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.use_skip_connections = use_skip_connections
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        if self.use_norm == "BatchNorm":
            self.batchnorm = nn.BatchNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif self.use_norm == "InstanceNorm":
            self.instancenorm = nn.InstanceNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif self.use_norm == "LayerNorm":
            self.layernorm = nn.LayerNorm([out_channels, z_dim_for_layernorm])
            # self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.5)

    def forward(self, x):
        x_orig = x
        x = self.conv(x)
        if self.use_norm == "BatchNorm":
            x = self.batchnorm(x)  # for BN1d
        elif self.use_norm == "InstanceNorm":
            x = self.instancenorm(x)
        elif self.use_norm == "LayerNorm":
            x = self.layernorm(x)
        elif self.use_norm == "None":
            x = x
        else:
            raise ValueError("unknown normalization")

        x = self.activation(x)

        if self.use_skip_connections and self.in_channels == self.out_channels:
            x = x + x_orig
        return x

class ConvKernelMixingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        padding_mode,
        use_norm,
        use_skip_connections,
        activation,
        momentum_for_batchnorm,
        z_dim_for_layernorm=40,
    ):
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.use_skip_connections = use_skip_connections
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")
        self.conv_kernel1 = nn.Conv1d(
            4*out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            padding_mode=padding_mode,
        )
        self.conv_kernel3 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.conv_kernel5 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            padding_mode=padding_mode,
        )
        self.conv_kernel7 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding=3,
            padding_mode=padding_mode,
        )
        self.conv_kernel9 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=9,
            padding=4,
            padding_mode=padding_mode,
        )
        if self.use_norm == "BatchNorm":
            self.batchnorm = nn.BatchNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif self.use_norm == "InstanceNorm":
            self.instancenorm = nn.InstanceNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif self.use_norm == "LayerNorm":
            self.layernorm = nn.LayerNorm([out_channels, z_dim_for_layernorm])
            # self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.5)

    def forward(self, x):
        x_orig = x
        #print(f"{x.device=}")
        #print(f"{self.conv_kernel3.weight.device=}")
        x3 = self.conv_kernel3(x)
        x5 = self.conv_kernel5(x)
        x7 = self.conv_kernel7(x)
        x9 = self.conv_kernel9(x)
        #print(f"{x.shape=}")
        x = torch.cat([x3, x5, x7, x9], dim=1)
        x = self.conv_kernel1(x)
        #print(f"{x.shape=}")
        if self.use_norm == "BatchNorm":
            x = self.batchnorm(x)  # for BN1d
        elif self.use_norm == "InstanceNorm":
            x = self.instancenorm(x)
        elif self.use_norm == "LayerNorm":
            x = self.layernorm(x)
        elif self.use_norm == "None":
            x = x
        else:
            raise ValueError("unknown normalization")

        x = self.activation(x)

        if self.use_skip_connections and self.in_channels == self.out_channels:
            x = x + x_orig
        return x


class ConvEncoderTimeSeries(nn.Module):
    def __init__(
        self,
        input_size,
        input_time_series_length,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
    ):
        super().__init__()
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        # print(f"{use_norm} is used as the normalization layer in ConvEncoderTimeSeries")
        assert activation in ["relu", "gelu", "None"]
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")
        padding = (kernel_size - 1) // 2
        self.input_size = input_size
        self.channels = channels
        self.num_layers = len(channels)
        # input channel of the first layer is 1 (fixed)
        channels = [input_time_series_length] + channels
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode="circular",
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    activation=activation,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    **extra_args,
                )
                for i in range(self.num_layers)
            ]
        )
        self.use_skip_connections = (
            use_skip_connections  # now only for the middle layers of same size
        )
        # if use_norm == "BatchNorm":
        #     print("batchnorm included in ConvEncoderTimeSeries")
        # elif use_norm == "InstanceNorm":
        #     print("instancenorm included in ConvEncoderTimeSeries")
        # elif use_norm == "LayerNorm":
        #     print("layernorm included in ConvEncoderTimeSeries")
        # if use_skip_connections:
        #     print("skip connection included in ConvEncoderTimeSeries")
        self.fc1 = nn.Linear(input_size * channels[-1], output_size)
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        for _i, l in enumerate(self.conv_blocks):
            # print(f"{_i}: l.batchnorm.running_mean: ", l.batchnorm.running_mean)
            # print(f"{_i}: l.batchnorm.running_var: ", l.batchnorm.running_var)
            if self.use_skip_connections and isinstance(l, ConvBlock):
                if l.in_channels == l.out_channels:
                    # print("skip connection")
                    x = self.activation(l(x)) + x
                else:
                    x = self.activation(l(x))
            else:
                x = self.activation(l(x))

        x = self.fc1(x.reshape(-1, self.input_size * self.channels[-1])).squeeze(1)
        return x

class ConvEncoderTimeSeriesKernelSizes(nn.Module):
    def __init__(
        self,
        input_size,
        input_time_series_length,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
    ):
        super().__init__()
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        # print(f"{use_norm} is used as the normalization layer in ConvEncoderTimeSeries")
        assert activation in ["relu", "gelu", "None"]
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "None":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError("unknown activation")
        padding = (kernel_size - 1) // 2
        self.input_size = input_size
        self.channels = channels
        self.num_layers = len(channels)
        # input channel of the first layer is 1 (fixed)
        channels = [input_time_series_length] + channels
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm

        self.conv_blocks = nn.ModuleList(
            [
                ConvKernelMixingBlock
                (
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    padding=padding,
                    padding_mode="circular",
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    activation=activation,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    **extra_args,
                )
                for i in range(self.num_layers)
            ]
        )
        self.use_skip_connections = (
            use_skip_connections  # now only for the middle layers of same size
        )
        # if use_norm == "BatchNorm":
        #     print("batchnorm included in ConvEncoderTimeSeries")
        # elif use_norm == "InstanceNorm":
        #     print("instancenorm included in ConvEncoderTimeSeries")
        # elif use_norm == "LayerNorm":
        #     print("layernorm included in ConvEncoderTimeSeries")
        # if use_skip_connections:
        #     print("skip connection included in ConvEncoderTimeSeries")
        self.fc1 = nn.Linear(input_size * channels[-1], output_size)
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        for _i, l in enumerate(self.conv_blocks):
            # print(f"{_i}: l.batchnorm.running_mean: ", l.batchnorm.running_mean)
            # print(f"{_i}: l.batchnorm.running_var: ", l.batchnorm.running_var)
            if self.use_skip_connections and isinstance(l, ConvBlock):
                if l.in_channels == l.out_channels:
                    # print("skip connection")
                    x = self.activation(l(x)) + x
                else:
                    x = self.activation(l(x))
            else:
                x = self.activation(l(x))

        x = self.fc1(x.reshape(-1, self.input_size * self.channels[-1])).squeeze(1)
        return x


class ConvEncoderTimeSeries_VariableKernelSize(nn.Module):
    def __init__(
        self,
        input_size,
        input_time_series_length,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
    ):
        super().__init__()
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        # print(f"{use_norm} is used as the normalization layer in ConvEncoderTimeSeries")
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
        self.channels = channels
        self.num_layers = len(channels)
        assert len(kernel_size) == len(channels)
        padding = [(k - 1) // 2 for k in kernel_size]
        # input channel of the first layer is 1 (fixed)
        channels = [input_time_series_length] + channels
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size[i],
                    padding=padding[i],
                    padding_mode="circular",
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    activation=activation,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    **extra_args,
                )
                for i in range(self.num_layers)
            ]
        )
        self.use_skip_connections = (
            use_skip_connections  # now only for the middle layers of same size
        )
        # if use_norm == "BatchNorm":
        #     print("batchnorm included in ConvEncoderTimeSeries")
        # elif use_norm == "InstanceNorm":
        #     print("instancenorm included in ConvEncoderTimeSeries")
        # elif use_norm == "LayerNorm":
        #     print("layernorm included in ConvEncoderTimeSeries")
        # if use_skip_connections:
        #     print("skip connection included in ConvEncoderTimeSeries")
        self.fc1 = nn.Linear(input_size * channels[-1], output_size)
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        for _i, l in enumerate(self.conv_blocks):
            # print(f"{_i}: l.batchnorm.running_mean: ", l.batchnorm.running_mean)
            # print(f"{_i}: l.batchnorm.running_var: ", l.batchnorm.running_var)
            if self.use_skip_connections and isinstance(l, ConvBlock):
                if l.in_channels == l.out_channels:
                    # print("skip connection")
                    x = self.activation(l(x)) + x
                else:
                    x = self.activation(l(x))
            else:
                x = self.activation(l(x))

        x = self.fc1(x.reshape(-1, self.input_size * self.channels[-1])).squeeze(1)
        return x


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
    ):
        assert use_norm in ["BatchNorm", "InstanceNorm", "LayerNorm", "None"]
        # print(f"{use_norm} is used as the normalization layer in ConvEncoder")
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
        padding = (kernel_size - 1) // 2
        self.input_size = input_size
        self.channels = channels
        self.num_layers = len(channels)
        # input channel of the first layer is 1 (fixed)
        channels = [1] + channels
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode="circular",
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    activation=activation,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    **extra_args,
                )
                for i in range(self.num_layers)
            ]
        )
        self.use_skip_connections = (
            use_skip_connections  # now only for the middle layers of same size
        )
        # print("use_norm: ", use_norm)
        # if use_norm == "BatchNorm":
        #     print("batchnorm included in ConvEncoder")
        # elif use_norm == "InstanceNorm":
        #     print("instancenorm included in ConvEncoder")
        # elif use_norm == "LayerNorm":
        #     print("layernorm included in ConvEncoder")
        # if use_skip_connections:
        #     print("skip connection included in ConvEncoder")
        self.fc1 = nn.Linear(input_size * channels[-1], output_size)
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        x = x.unsqueeze(1)
        for _i, l in enumerate(self.conv_blocks):
            x = l(x)

        x = self.fc1(x.reshape(-1, self.input_size * self.channels[-1])).squeeze(1)
        return x

class ConvDecoder_VariableKernelSize(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        padding_mode,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
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
        self.num_layers = len(channels)
        assert len(kernel_size) == len(channels)
        channels = list(reversed([1] + channels))  # Encoderと逆順
        kernel_size = list(reversed(kernel_size)) # Encoderと逆順
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm

        self.blocks = nn.ModuleList(
            [
                ConvTransposeBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size[i],
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation=activation,
                    padding_mode=padding_mode,
                    **extra_args,
                )
                for i in range(self.num_layers - 1)
            ]
            + [
                ConvTransposeBlock(
                    channels[self.num_layers - 1],
                    channels[self.num_layers],
                    kernel_size=kernel_size[self.num_layers-1],
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation="None",  # relu is not applied to the last layer
                    padding_mode=padding_mode,
                    **extra_args,
                )
            ]
        )

        self.channels = channels
        self.fc1 = nn.Linear(input_size, output_size * self.channels[0])
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        x = self.fc1(x)
        x = rearrange(x, "b (c h) -> b c h", c=self.channels[0], h=self.output_size)
        for _i, l in enumerate(self.blocks):
            #print(f"{_i=}, {l=}, {x.shape=}")
            x = l(x)

        x = x.squeeze(1)
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        padding_mode,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
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
        self.num_layers = len(channels)
        channels = list(reversed([1] + channels)) # Encoderと逆順
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm

        self.blocks = nn.ModuleList(
            [
                ConvTransposeBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation=activation,
                    padding_mode=padding_mode,
                    **extra_args,
                )
                for i in range(self.num_layers - 1)
            ]
            + [
                ConvTransposeBlock(
                    channels[self.num_layers - 1],
                    channels[self.num_layers],
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation="None",  # relu is not applied to the last layer
                    padding_mode=padding_mode,
                    **extra_args,
                )
            ]
        )

        self.channels = channels
        self.fc1 = nn.Linear(input_size, output_size * self.channels[0])
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        x = self.fc1(x)
        x = rearrange(x, "b (c h) -> b c h", c=self.channels[0], h=self.output_size)
        for _i, l in enumerate(self.blocks):
            x = l(x)

        x = x.squeeze(1)
        return x

class ConvDecoderKernelSizes(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        padding_mode,
        input_normalize_factor,
        activation,
        z_dim_for_layernorm,
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
        self.num_layers = len(channels)
        channels = list(reversed([1] + channels)) # Encoderと逆順
        extra_args = {}
        if use_norm == "LayerNorm":
            extra_args["z_dim_for_layernorm"] = z_dim_for_layernorm

        self.blocks = nn.ModuleList(
            [
                ConvTransposeKernelMixingBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation=activation,
                    padding_mode=padding_mode,
                    **extra_args,
                )
                for i in range(self.num_layers - 1)
            ]
            + [
                ConvTransposeKernelMixingBlock(
                    channels[self.num_layers - 1],
                    channels[self.num_layers],
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                    use_skip_connections=use_skip_connections,
                    momentum_for_batchnorm=momentum_for_batchnorm,
                    activation="None",  # relu is not applied to the last layer
                    padding_mode=padding_mode,
                    **extra_args,
                )
            ]
        )

        self.channels = channels
        self.fc1 = nn.Linear(input_size, output_size * self.channels[0])
        self.input_normalize_factor = input_normalize_factor

    def forward(self, x):
        x = x * self.input_normalize_factor
        x = self.fc1(x)
        x = rearrange(x, "b (c h) -> b c h", c=self.channels[0], h=self.output_size)
        for _i, l in enumerate(self.blocks):
            x = l(x)

        x = x.squeeze(1)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        activation,
        padding_mode,
        z_dim_for_layernorm=40,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.use_skip_connections = use_skip_connections
        self.conv_transpose = MyConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
        )
        if use_norm == "BatchNorm":
            self.batchnorm = nn.BatchNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif use_norm == "InstanceNorm":
            self.instancenorm = nn.InstanceNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif use_norm == "LayerNorm":
            self.layernorm = nn.LayerNorm([out_channels, z_dim_for_layernorm])
            # self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.0001)
        else:
            raise ValueError("unknown normalization method")

        assert activation in ["relu", "gelu", "None"]
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
        x = self.conv_transpose(x)
        if self.use_norm == "BatchNorm":
            x = self.batchnorm(x)  # for BN1d
        elif self.use_norm == "InstanceNorm":
            x = self.instancenorm(x)  # for BN1d
        elif self.use_norm == "LayerNorm":
            x = self.layernorm(x)
        elif self.use_norm == "None":
            x = x
        x = self.activation(x)

        if self.use_skip_connections and self.in_channels == self.out_channels:
            x = x + x_orig
        return x

class ConvTransposeKernelMixingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_norm,
        use_skip_connections,
        momentum_for_batchnorm,
        activation,
        padding_mode,
        z_dim_for_layernorm=40,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.use_skip_connections = use_skip_connections
        self.conv1 = nn.Conv1d(
            4*out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            padding_mode=padding_mode,
        )
        self.conv_transpose3 = MyConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding_mode=padding_mode,
        )
        self.conv_transpose5 = MyConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding_mode=padding_mode,
        )
        self.conv_transpose7 = MyConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding_mode=padding_mode,
        )
        self.conv_transpose9 = MyConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=9,
            padding_mode=padding_mode,
        )
        if use_norm == "BatchNorm":
            self.batchnorm = nn.BatchNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif use_norm == "InstanceNorm":
            self.instancenorm = nn.InstanceNorm1d(
                out_channels, momentum=momentum_for_batchnorm
            )
        elif use_norm == "LayerNorm":
            self.layernorm = nn.LayerNorm([out_channels, z_dim_for_layernorm])
            # self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.0001)
        else:
            raise ValueError("unknown normalization method")

        assert activation in ["relu", "gelu", "None"]
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
        x3 = self.conv_transpose3(x)
        x5 = self.conv_transpose5(x)
        x7 = self.conv_transpose7(x)
        x9 = self.conv_transpose9(x)
        #print(f"{x.shape=}")
        x = torch.cat([x3, x5, x7, x9], dim=-2)
        x = self.conv1(x)
        #print(f"{x.shape=}")
        #x = self.conv_transpose(x)
        if self.use_norm == "BatchNorm":
            x = self.batchnorm(x)  # for BN1d
        elif self.use_norm == "InstanceNorm":
            x = self.instancenorm(x)  # for BN1d
        elif self.use_norm == "LayerNorm":
            x = self.layernorm(x)
        elif self.use_norm == "None":
            x = x
        x = self.activation(x)

        if self.use_skip_connections and self.in_channels == self.out_channels:
            x = x + x_orig
        return x


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


def init_weights(m, mode="kaiming_uniform", bias=0.0):
    assert mode in [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
    ]
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, MyConvTranspose1d)
        or isinstance(m, nn.Linear)
        # or isinstance(m, ShiftedWeightLayer)
        # or isinstance(m, ShiftedWeightShrinkLayer)
        # or isinstance(m, ShiftedWeightExpandLayer)
    ):
        if mode == "xavier_uniform":
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(bias)
        elif mode == "xavier_normal":
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(bias)
        elif mode == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5), nonlinearity="relu")
            m.bias.data.fill_(bias)
        elif mode == "kaiming_normal":
            torch.nn.init.kaiming_normal_(m.weight, a=np.sqrt(5), nonlinearity="relu")
            m.bias.data.fill_(bias)


def network_init_conv_timeseries(
    input_dim,
    output_dim,
    hidden_dim,
    time_series_length,
    channels: List[int],
    kernel_size=5,
    weight_init="kaiming_uniform",
    bias=0.0,
    use_norm="LayerNorm",
    use_skip_connections=False,
    momentum_for_batchnorm=0.1,
    activation="relu",
    input_normalize_factor=1.0,
):
    encoder = ConvEncoderTimeSeries(
        input_dim,
        time_series_length,
        hidden_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        input_normalize_factor=input_normalize_factor,
        z_dim_for_layernorm=input_dim,
    )
    decoder = ConvDecoder(
        hidden_dim,
        output_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        padding_mode="circular",
        input_normalize_factor=1.0,
        z_dim_for_layernorm=output_dim,
    )
    encoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    decoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    lambdas = torch.rand(hidden_dim) * 0.01  # 231127 original
    # lambdas = torch.log(torch.rand(hidden_dim)+0.5) # 231127 for demo
    # lambdas = lambdas
    return encoder, decoder, lambdas


def network_init_Lorenz96(time_sequence_length, z_dim=40, kernel=3, G_nondiag=False):
    random_seed = 42
    x_dim = 40
    hidden_size = 40
    time_sequence_length = time_sequence_length
    if random_seed != 0:
        # print(f"manual seed for BayesFaithfulFilter, {random_seed=}")
        torch.manual_seed(random_seed)
    channels = [20, 20, 20, 20, 20, 20, 20, 20]
    #channels = [20, 20, 20, 20, 20, 10, 5]
    if G_nondiag:
        encoder1, encoder2, decoder, lambdas = network_init_conv_time_series_Gnondiag(
            input_dim=x_dim,
            output_dim=x_dim,
            time_series_length=time_sequence_length,
            hidden_dim1=z_dim,
            hidden_dim2=(z_dim//2)*3,
            channels=channels,
            kernel_size=kernel,
            weight_init="kaiming_normal",
            bias=0.0,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
            input_normalize_factor=1.0,
        )
        # print(f"{z_dim=}")
        # print(f"{hidden_size=}")
        f_network = copy.deepcopy(encoder1).cuda()#to(device)
        G_network = copy.deepcopy(encoder2).cuda()#.to(device)
    else:
        encoder, decoder, lambdas = network_init_conv_timeseries(
            input_dim=x_dim,
            output_dim=x_dim,
            time_series_length=time_sequence_length,
            hidden_dim=z_dim,
            channels=channels,
            kernel_size=kernel,
            weight_init="kaiming_normal",
            bias=0.0,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
            input_normalize_factor=1.0,
        )
        # print(f"{z_dim=}")
        # print(f"{hidden_size=}")
        f_network = copy.deepcopy(encoder).cuda()#.to(device)
        G_network = copy.deepcopy(encoder).cuda()#.to(device)
    h_original = decoder.cuda()#.to(device)
    h_network = h_original
    lambdas = lambdas
    lambdas.requires_grad = True
    add_factor = 10000 # need to modify together with the last channel dimension
    to_add = add_factor * torch.ones(G_network.fc1.out_features).cuda()#.to(device)
    G_network.fc1.bias = nn.Parameter(G_network.fc1.bias + to_add)
    # print(f_network)
    return f_network, G_network, h_network, lambdas

def network_init_Lorenz96_kernelsizes(time_sequence_length, z_dim=40, G_nondiag=False):
    random_seed = 42
    x_dim = 40
    hidden_size = 40
    time_sequence_length = time_sequence_length
    if random_seed != 0:
        # print(f"manual seed for BayesFaithfulFilter, {random_seed=}")
        torch.manual_seed(random_seed)
    channels = [50, 50, 50, 50, 50, 50]
    #channels = [5, 5, 5]
    #kernel_size = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    if G_nondiag:
        encoder1, encoder2, decoder, lambdas = network_init_conv_time_series_kernelsizes(
            input_dim=x_dim,
            output_dim=x_dim,
            time_series_length=time_sequence_length,
            hidden_dim1=z_dim,
            hidden_dim2=(z_dim//2)*3,
            channels=channels,
            kernel_size=kernel_size,
            weight_init="kaiming_normal",
            bias=0.0,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
            input_normalize_factor=1.0,
        )
        # print(f"{z_dim=}")
        # print(f"{hidden_size=}")
        f_network = copy.deepcopy(encoder1).cuda()#to(device)
        G_network = copy.deepcopy(encoder2).cuda()#.to(device)
    else:
        encoder, decoder, lambdas = network_init_conv_time_series_kernelsizes(
            input_dim=x_dim,
            output_dim=x_dim,
            time_series_length=time_sequence_length,
            hidden_dim=z_dim,
            channels=channels,
            kernel_size=3,
            weight_init="kaiming_normal",
            bias=0.0,
            use_norm="LayerNorm",
            use_skip_connections=True,
            momentum_for_batchnorm=0.01,
            activation="relu",
            input_normalize_factor=1.0,
        )
        # print(f"{z_dim=}")
        # print(f"{hidden_size=}")
        f_network = copy.deepcopy(encoder).cuda()#.to(device)
        G_network = copy.deepcopy(encoder).cuda()#.to(device)
    h_original = decoder.cuda()#.to(device)
    h_network = h_original
    lambdas = lambdas
    lambdas.requires_grad = True
    # print("G_network.fc1.bias: ", G_network.fc1.bias)
    add_factor = 1000
    to_add = add_factor * torch.ones(G_network.fc1.out_features).cuda()#.to(device)
    G_network.fc1.bias = nn.Parameter(G_network.fc1.bias + to_add)
    # print("G_network.fc1.bias: ", G_network.fc1.bias)
    # print(f_network)
    return f_network, G_network, h_network, lambdas

def network_init_Lorenz96_transformer(time_sequence_length, z_dim=40):
    random_seed = 42
    x_dim = 40
    hidden_size = 40
    time_sequence_length = time_sequence_length
    if random_seed != 0:
        # print(f"manual seed for BayesFaithfulFilter, {random_seed=}")
        torch.manual_seed(random_seed)
    channels = [50, 50, 50, 50, 50, 50, 50, 20, 10, 5]
    kernel_size = [3, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    encoder, decoder, lambdas = network_init_time_series_transformer(
        input_dim=x_dim,
        output_dim=x_dim,
        time_series_length=time_sequence_length,
        hidden_dim=z_dim,
        channels=channels,
        kernel_size=kernel_size,
        weight_init="kaiming_normal",
        bias=0.0,
        use_norm="LayerNorm",
        use_skip_connections=True,
        momentum_for_batchnorm=0.01,
        activation="relu",
        input_normalize_factor=1.0,
    )
    # print(f"{z_dim=}")
    # print(f"{hidden_size=}")
    f_network = copy.deepcopy(encoder).cuda()#.to(device)
    G_network = copy.deepcopy(encoder).cuda()#.to(device)
    h_original = decoder.cuda()#.to(device)
    h_network = h_original
    lambdas = lambdas
    lambdas.requires_grad = True
    # print("G_network.fc1.bias: ", G_network.fc1.bias)
    add_factor = 10
    to_add = add_factor * torch.ones(G_network.fc1.out_features).cuda()#.to(device)
    G_network.fc1.bias = nn.Parameter(G_network.fc1.bias + to_add)
    # print("G_network.fc1.bias: ", G_network.fc1.bias)
    # print(f_network)
    return f_network, G_network, h_network, lambdas


def network_init_conv_time_series_Gnondiag(
    input_dim,
    output_dim,
    hidden_dim1,
    hidden_dim2,
    time_series_length,
    channels: List[int],
    kernel_size=5,
    weight_init="kaiming_uniform",
    bias=0.0,
    use_norm="LayerNorm",
    use_skip_connections=False,
    momentum_for_batchnorm=0.1,
    activation="relu",
    input_normalize_factor=1.0,
    G_nondiag=False,
):
    # if obs_data_complete:
    encoder1 = ConvEncoderTimeSeries(
        input_dim,
        time_series_length,
        hidden_dim1,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        input_normalize_factor=input_normalize_factor,
        z_dim_for_layernorm=input_dim,
    )
    encoder2 = ConvEncoderTimeSeries(
        input_dim,
        time_series_length,
        hidden_dim2,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        input_normalize_factor=input_normalize_factor,
        z_dim_for_layernorm=input_dim,
    )
    decoder = ConvDecoder(
        hidden_dim1,
        output_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        padding_mode="circular",
        input_normalize_factor=1.0,
        z_dim_for_layernorm=output_dim,
    )
    encoder1.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    encoder2.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    decoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    lambdas = torch.rand(hidden_dim1) * 0.01
    # lambdas = lambdas
    return encoder1, encoder2, decoder, lambdas

def network_init_conv_time_series(
    input_dim,
    output_dim,
    hidden_dim,
    time_series_length,
    channels: List[int],
    kernel_size=5,
    weight_init="kaiming_uniform",
    bias=0.0,
    use_norm="LayerNorm",
    use_skip_connections=False,
    momentum_for_batchnorm=0.1,
    activation="relu",
    input_normalize_factor=1.0,
    G_nondiag=False,
):
    # if obs_data_complete:
    encoder = ConvEncoderTimeSeries(
        input_dim,
        time_series_length,
        hidden_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        input_normalize_factor=input_normalize_factor,
        z_dim_for_layernorm=input_dim,
    )
    decoder = ConvDecoder(
        hidden_dim,
        output_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        padding_mode="circular",
        input_normalize_factor=1.0,
        z_dim_for_layernorm=output_dim,
    )
    encoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    decoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    lambdas = torch.rand(hidden_dim) * 0.01
    # lambdas = lambdas
    return encoder, decoder, lambdas


def network_init_conv_time_series_kernelsizes(
    input_dim,
    output_dim,
    hidden_dim,
    time_series_length,
    channels: List[int],
    kernel_size: List[int],
    weight_init="kaiming_uniform",
    bias=0.0,
    use_norm="LayerNorm",
    use_skip_connections=False,
    momentum_for_batchnorm=0.1,
    activation="relu",
    input_normalize_factor=1.0,
    G_nondiag=False,
):
    # if obs_data_complete:
    encoder = ConvEncoderTimeSeriesKernelSizes(
        input_dim,
        time_series_length,
        hidden_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        input_normalize_factor=input_normalize_factor,
        z_dim_for_layernorm=input_dim,
    )
    decoder = ConvDecoderKernelSizes(
        hidden_dim,
        output_dim,
        channels,
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_skip_connections=use_skip_connections,
        momentum_for_batchnorm=momentum_for_batchnorm,
        activation=activation,
        padding_mode="circular",
        input_normalize_factor=1.0,
        z_dim_for_layernorm=output_dim,
    )
    encoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    decoder.apply(lambda m: init_weights(m, mode=weight_init, bias=bias))
    lambdas = torch.rand(hidden_dim) * 0.01
    # lambdas = lambdas
    return encoder, decoder, lambdas


# experimental
def network_init_ViT(
    input_dim,
    output_dim,
    hidden_dim,
    D,
    patch_size_encoder,
    patch_size_decoder,
    num_layers=6,
    num_heads=8,
    dim_feedforward=512,
    activation="gelu",
    norm_first=False,
    input_normalize_factor=1.0,
):
    assert input_dim % patch_size_encoder == 0
    assert hidden_dim % patch_size_decoder == 0
    # if obs_data_complete:
    encoder = ViT_encoder_for_Lorenz96(
        x_dim=input_dim,
        patch_size=patch_size_encoder,
        D=D,
        hidden_dim=hidden_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        activation=activation,
        norm_first=norm_first,
        input_normalize_factor=input_normalize_factor,
    )
    decoder = ViT_encoder_for_Lorenz96(
        x_dim=hidden_dim,
        patch_size=patch_size_decoder,
        D=D,
        hidden_dim=output_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        activation=activation,
        norm_first=norm_first,
        input_normalize_factor=1.0,
    )
    lambdas = torch.rand(hidden_dim) * 0.01
    return encoder, decoder, lambdas


# experimental
def network_init_ViT_TimeSeries(
    input_dim,
    output_dim,
    hidden_dim,
    D,
    patch_size_encoder,
    patch_size_decoder,
    time_series_length,
    num_layers=6,
    num_heads=8,
    dim_feedforward=512,
    activation="gelu",
    norm_first=False,
    input_normalize_factor=1.0,
):
    # if obs_data_complete:
    encoder = ViT_encoder_for_Lorenz96_TimeSeries(
        x_dim=input_dim,
        patch_size=patch_size_encoder,
        time_series_length=time_series_length,
        D=D,
        hidden_dim=hidden_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        activation=activation,
        norm_first=norm_first,
        input_normalize_factor=input_normalize_factor,
    )
    decoder = ViT_encoder_for_Lorenz96(
        x_dim=hidden_dim,
        patch_size=patch_size_decoder,
        D=D,
        hidden_dim=output_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        activation=activation,
        norm_first=norm_first,
        input_normalize_factor=1.0,
    )
    return encoder, decoder
