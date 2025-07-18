# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import numpy as np

import torch
import torch.nn as nn
from einops import repeat
from timm.models.vision_transformer import Attention, Mlp


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: int (M) or a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if isinstance(pos, int):
        pos = np.arange(pos, dtype=np.float32)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TransformerBlock(nn.Module):
    """ A transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_cond=False,
        **block_kwargs
    ):
        super().__init__()
        self.use_cond = use_cond

        if self.use_cond:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c=None):
        if self.use_cond:
            assert c is not None, 'provide condition'
            (
                shift_msa, scale_msa, gate_msa,
                shift_mlp, scale_mlp, gate_mlp
            ) = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))

        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        out_channels,
        use_cond=False,
        head_type='mlp',
        drop_prob=0.0
    ):
        super().__init__()
        self.use_cond = use_cond
        self.head_type = head_type

        if self.use_cond:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        else:
            if head_type == 'simple':
                self.linear = nn.Linear(hidden_size, out_channels, bias=True)
            elif head_type == 'mlp':
                self.linear = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.Sequential( # for compatibility with old checkpoints
                        nn.Dropout() if drop_prob > 0 else nn.Identity(),
                        nn.SiLU(),
                    ),
                    nn.Linear(hidden_size, out_channels, bias=True),
                )
            else:
                raise ValueError(f'unknown head_type={head_type}')
    
    def initialize_weights(self):
        # Zero-out output layers
        if self.use_cond:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        if self.use_cond or self.head_type == 'simple':
            nn.init.constant_(self.linear.weight, 0)
            nn.init.constant_(self.linear.bias, 0)
        else:
            nn.init.constant_(self.linear[-1].weight, 0)
            nn.init.constant_(self.linear[-1].bias, 0)


    def forward(self, x, c=None):
        if self.use_cond:
            assert c is not None, 'provide condition'
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)

        x = self.linear(x)
        return x


class TransformerHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_size=256,
        max_seq_len=240,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        use_cond=False,
        head_type='simple',
        train_pos_embed=False,
        drop_prob=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.use_cond = use_cond
        self.train_pos_embed = train_pos_embed

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len + 1, hidden_size),
            requires_grad=bool(self.train_pos_embed)
        )

        if self.use_cond:
            self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio, use_cond=self.use_cond
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size, self.out_channels,
            use_cond=self.use_cond,
            head_type=head_type,
            drop_prob=drop_prob
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if not self.train_pos_embed:
            pos_embed = get_1d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.pos_embed.shape[1]
            )
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )

        if self.use_cond:
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        self.final_layer.initialize_weights()

    def forward(self, x, t=None):
        """
        x: (N, C, T) tensor of temporal inputs
        t: (N,) tensor of inter-frame distances
        """
        b, _, _ = x.shape
    
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.shape[1]]        # (N, T, D)
        c = (
            self.t_embedder(t) if self.use_cond else
            None                      # (N, D)
        )

        for block in self.blocks:
            x = block(x, c)           # (N, T, D)

        x = x[:, 0] # extract cls     # (N, D)
        pred = self.final_layer(x, c) # (N, D_out)
        return pred
