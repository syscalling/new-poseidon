
"""
This file contains scOT.

A lot of this file is taken from the transformers library and changed to our purposes. Huggingface Transformers is licensed under
Apache 2.0 License, see trainer.py for details.

We follow https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py
and https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129

The class ConvNeXtBlock is taken from the facebookresearch/ConvNeXt repository and is licensed under the MIT License,

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from transformers import (
    Swinv2PreTrainedModel,
    PretrainedConfig,
)
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2EncoderOutput,
    Swinv2Attention,
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    window_reverse,
    window_partition,
)
from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Union, Tuple, List
import math
import collections


@dataclass
class ScOTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ScOTConfig(PretrainedConfig):
    """https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py"""

    model_type = "swinv2"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        num_out_channels=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        skip_connections=[True, True, True],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,  # for loss: 1 for l1, 2 for l2
        channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
        residual_model="convnext",  # "convnext" or "resnet"
        use_conditioning=False,
        learn_residual=False,  # learn the residual for time-dependent problems
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.use_conditioning = use_conditioning
        self.learn_residual = learn_residual if self.use_conditioning else False
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # we set the hidden_size attribute in order to make Swinv2 work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.pretrained_window_sizes = (0, 0, 0, 0)
        self.num_out_channels = num_out_channels
        self.p = p
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, time):
        return super().forward(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Linear(1, dim)
        self.bias = nn.Linear(1, dim)

    def forward(self, x, time):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x**2).mean(dim=-1, keepdim=True) - mean**2
        x = (x - mean) / (var + self.eps).sqrt()
        time = time.reshape(-1, 1).type_as(x)
        weight = self.weight(time).unsqueeze(1)
        bias = self.bias(time).unsqueeze(1)
        if x.dim() == 4:
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return weight * x + bias


class ConvNeXtBlock(nn.Module):
    r"""Taken from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, config, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        if config.use_conditioning:
            layer_norm = ConditionalLayerNorm
        else:
            layer_norm = LayerNorm
        self.norm = layer_norm(dim, eps=config.layer_norm_eps)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.weight = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  # was gamma before
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        #! assumes square images
        input_dim = math.floor(sequence_length**0.5)

        input = x
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x, time)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape(batch_size, sequence_length, hidden_size)

        x = input + self.drop_path(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        kernel_size = 3
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        #! assumes square images
        input_dim = math.floor(sequence_length**0.5)

        input = x
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, sequence_length, hidden_size)
        x = x + input
        return x


class ScOTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )