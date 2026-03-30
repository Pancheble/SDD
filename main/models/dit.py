"""
models/dit.py
Diffusion Transformer (DiT-B style) with intermediate feature extraction.
논문 Section 3.3.2: DiT의 경우 중간 트랜스포머 블록에서 특징 추출.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), 'b c h w -> b (h w) c')


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block"""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )
        # AdaLN modulation: scale, shift for norm1, norm2, gate for attn, mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # attention
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa[:, None] * attn_out
        # mlp
        h = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size ** 2 * out_channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return self.linear(x)


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT-B/4).
    forward(x, t, return_features=True) 시
    (noise_pred, mid_features) 반환.

    논문 Section 3.3.2:
      feature_layers에 지정된 블록의 출력을 평균 pooling하여 특징 벡터 생성.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        feature_layers: list = None,   # 특징 추출할 블록 인덱스
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.input_size = input_size
        self.feature_layers = feature_layers or list(range(depth // 2 - 2, depth // 2 + 2))

        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        num_patches = (input_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.hidden_size = hidden_size

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, p²·C] → [B, C, H, W]"""
        p = self.patch_size
        h = w = self.input_size // p
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], self.in_channels, h * p, w * p)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, return_features: bool = False
    ):
        """
        Args:
            x: noisy image [B, C, H, W]
            t: timestep [B]
            return_features: True이면 (noise_pred, feature_vec) 반환

        Returns:
            noise_pred: [B, C, H, W]
            feature_vec (optional): [B, hidden_size]  (feature_layers 평균)
        """
        x = self.patch_embed(x) + self.pos_embed          # [B, N, D]
        c = self.t_embedder(t)                             # [B, D]

        mid_feats = []
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if i in self.feature_layers:
                # 패치 평균 풀링 → 특징 벡터
                mid_feats.append(x.mean(dim=1))           # [B, D]

        noise_pred = self.unpatchify(self.final_layer(x, c))

        if return_features:
            # feature_layers 출력 평균
            feature_vec = torch.stack(mid_feats, dim=0).mean(dim=0)  # [B, D]
            return noise_pred, feature_vec

        return noise_pred
