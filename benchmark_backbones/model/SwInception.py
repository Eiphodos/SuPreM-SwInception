# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Union

import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
from math import ceil
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from torch import nn, Tensor


class SwInception(nn.Module):
    """
    SwInception: "David Hagerman et al., "SwInception - Local Attention meets convolutions".
    Codebase inspired by and based on: "Yucheng et al.,
    Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int] = [96, 96, 96],
        hidden_size: int = 48,
        patch_size: Union[Sequence[int], int] = [2, 2, 2],
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        decode_hs_ratio: float = 2.0
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")


        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.img_size = img_size
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.hidden_size = hidden_size
        self.decode_hidden_size = int(hidden_size * decode_hs_ratio)
        self.classification = False
        self.encoder = SwInceptionEncoder(pretrain_img_size=self.img_size,
                                          patch_size=self.patch_size,
                                          embed_dim=self.hidden_size)
        self.fl_out_size = tuple(img_d // (p_d*(2**4)) for img_d, p_d in zip(self.img_size, self.patch_size))

        self.unet_encoders = nn.ModuleList()
        self.unet_decoders = nn.ModuleList()

        encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        decoder0 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.decode_hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            upsample_kernel_size=self.patch_size,
            norm_name=norm_name,
            res_block=True,
        )

        decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.decode_hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.unet_encoders.append(encoder0)
        self.unet_encoders.append(encoder1)
        self.unet_encoders.append(encoder2)
        self.unet_decoders.append(decoder0)
        self.unet_decoders.append(decoder1)

        for i_layer in range(1, self.encoder.num_layers):
            enc = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size * 2 ** i_layer,
                out_channels=self.decode_hidden_size * 2 ** i_layer,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )
            self.unet_encoders.append(enc)

            dec = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.decode_hidden_size * 2 ** i_layer,
                out_channels=self.decode_hidden_size * 2 ** (i_layer-1),
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
            self.unet_decoders.append(dec)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.decode_hidden_size, out_channels=out_channels)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        z = self.encoder(x_in)

        x = self.unet_decoders[-1](self.unet_encoders[-1](z[-1]), self.unet_encoders[-2](z[-2]))
        for i in range(1, self.encoder.num_layers):
            enc = self.unet_encoders[-(i+2)]
            dec = self.unet_decoders[-(i+1)]
            x = dec(x, enc(z[-(i+2)]))
        x = self.unet_decoders[0](x, self.unet_encoders[0](x_in))
        return self.out(x)


class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,2,2).
        in_chans (int): Number of input volume channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 48.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, vol_size=(96, 96, 96), patch_size=(2, 2, 2), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        vol_size = ensure_tuple_rep(vol_size, 3)
        patch_size = ensure_tuple_rep(patch_size, 3)
        patches_resolution = [vol_size[0] // patch_size[0], vol_size[1] // patch_size[1], vol_size[2] // patch_size[2]]
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, D, Wh, Ww)

        return x


class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


class BasicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn=nn.GELU, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)
        self.act = act_fn()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Inception1x1(nn.Module):
    def __init__(self, in_features, out_features, **kwargs: Any) -> None:
        super().__init__()
        self.branch1x1 = BasicConv3d(in_features, out_features, kernel_size=1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        return branch1x1

class Inception3x3(nn.Module):
    def __init__(self, in_features, out_features, bottleneck_divisor=8, **kwargs: Any) -> None:
        super().__init__()
        bn_dim = in_features // bottleneck_divisor
        self.branch3x3_1 = BasicConv3d(in_features, bn_dim, kernel_size=1, **kwargs)
        self.branch3x3_2 = BasicConv3d(bn_dim, out_features, kernel_size=3, padding=1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        return branch3x3

class Inception5x5(nn.Module):
    def __init__(self, in_features, out_features, bottleneck_divisor=8, **kwargs: Any) -> None:
        super().__init__()
        bn_dim = in_features // bottleneck_divisor
        self.branch3x3dbl_1 = BasicConv3d(in_features, bn_dim, kernel_size=1, **kwargs)
        self.branch3x3dbl_2 = BasicConv3d(bn_dim, bn_dim, kernel_size=3, padding=1, **kwargs)
        self.branch3x3dbl_3 = BasicConv3d(bn_dim, out_features, kernel_size=3, padding=1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        return branch3x3dbl


class Inception7x7(nn.Module):
    def __init__(self, in_features, out_features, bottleneck_divisor=8, **kwargs: Any) -> None:
        super().__init__()
        bn_dim = in_features // bottleneck_divisor
        self.branch3x3trpl_1 = BasicConv3d(in_features, bn_dim, kernel_size=1, **kwargs)
        self.branch3x3trpl_2 = BasicConv3d(bn_dim, bn_dim, kernel_size=3, padding=1, **kwargs)
        self.branch3x3trpl_3 = BasicConv3d(bn_dim, bn_dim, kernel_size=3, padding=1, **kwargs)
        self.branch3x3trpl_4 = BasicConv3d(bn_dim, out_features, kernel_size=3, padding=1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3dbl = self.branch3x3trpl_1(x)
        branch3x3dbl = self.branch3x3trpl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3trpl_3(branch3x3dbl)
        branch3x3dbl = self.branch3x3trpl_4(branch3x3dbl)
        return branch3x3dbl


class InceptionPool(nn.Module):
    def __init__(self, in_features, out_features, **kwargs: Any) -> None:
        super().__init__()
        self.branch_pool_1 = nn.AvgPool3d(kernel_size=3, stride=1, padding=1, **kwargs)
        self.branch_pool_2 = BasicConv3d(in_features, out_features, kernel_size=1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)
        return branch_pool


class InceptionHead(nn.Module):
    '''Inception head used for Swinception'''

    def __init__(self, in_features, input_resolution, hidden_features=None, out_features=None, drop=0.,
                 branch_weights=[1, 1, 1, 1]):
        super().__init__()
        self.out_features = out_features or in_features
        self.input_resolution = input_resolution

        n_branches = len(branch_weights)
        hf = hidden_features or in_features

        inception_blocks = [Inception1x1, Inception3x3, Inception5x5, InceptionPool]
        assert len(inception_blocks) == len(branch_weights)

        block_weights = np.array(branch_weights)
        norm_block_weights = (1 / sum(block_weights)) * block_weights
        self.branch_dims = [int(hf * norm_block_weights[b]) for b in range(n_branches)]

        self.hidden_features = sum(self.branch_dims)

        self.branches = nn.ModuleList()
        for b in range(n_branches):
            self.branches.append(inception_blocks[b](in_features, self.branch_dims[b]))

        self.fc = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def _forward(self, x: Tensor) -> List[Tensor]:
        outputs = []
        for b in self.branches:
            outputs.append(b(x))
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        # Reshape input into 3D volume
        B, L, C = x.shape
        S, H, W = self.input_resolution
        x = x.permute(0, 2, 1).reshape(B, C, S, H, W).contiguous()

        # Apply inception layers
        x = self._forward(x)
        x = torch.cat(x, 1)

        # Reshape back into Transformer tokens
        x = x.reshape(B, self.hidden_features, S * H * W).permute(0, 2, 1).contiguous()

        # Apply final linear layer
        x = self.fc(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 n_windows=0, use_rel_pos_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.n_windows = n_windows
        self.n_attn_tokens = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.use_rel_pos_bias = use_rel_pos_bias

        if self.use_rel_pos_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                            num_heads))

            # get pair-wise relative position index for each token inside the window
            coords_s = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()

            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_embed=None):

        B_, N, C = x.shape
        n_attn_tokens = self.window_size[0] * self.window_size[1] * self.window_size[2]

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() # (3, B_, num_heads, N, C // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        if self.use_rel_pos_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                n_attn_tokens,
                n_attn_tokens, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rel_pos_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        n_windows = ceil(self.input_resolution[0] / self.window_size) * ceil(self.input_resolution[1] / self.window_size) * \
                    ceil(self.input_resolution[2] / self.window_size)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads, use_rel_pos_bias=use_rel_pos_bias,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_windows=n_windows)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = InceptionHead(in_features=dim, input_resolution=input_resolution, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, mask_matrix):

        B, L, C = x.shape
        S, H, W = self.input_resolution

        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, pos_embed=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)

        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 2 * C)

        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 use_rel_pos_bias=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        # build blocks

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_rel_pos_bias=use_rel_pos_bias)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):

        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class SwInceptionEncoder(nn.Module):

    def __init__(self,
                 pretrain_img_size=[96, 96, 96],
                 patch_size=[2, 2, 2],
                 in_chans=1,
                 embed_dim=48,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 7, 7],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 use_abs_pos_emb=False,
                 use_rel_pos_bias=True
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        # split image into non-overlapping patches

        pe_dim = embed_dim
        self.patch_embed = PatchEmbed3D(
            vol_size=pretrain_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=pe_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim), requires_grad=False)
        else:
            self.pos_embed = None
        if self.pos_embed is not None:
            pos_embed = get_3d_sincos_pos_embed(embed_dim, self.patch_embed.patches_resolution)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            #trunc_normal_(self.pos_embed, std=.02)


        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_rel_pos_bias=use_rel_pos_bias
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained), strict=False)
        else:
            self.apply(self._init_weights)

    def forward(self, input):
        """Forward function."""

        output = []

        x = self.patch_embed(input)

        output.append(x)

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)

        x = x.flatten(2).transpose(1, 2).contiguous()

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            if i in self.out_indices:
                x_out = F.layer_norm(x_out, [self.num_features[i]])

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                output.append(out)
        return output