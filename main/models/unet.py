"""
models/unet.py
ADM-style UNet with intermediate feature extraction support.
논문 Section 3.3.1 / 3.3.2 구현
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 기본 빌딩 블록
# ─────────────────────────────────────────────────────────────────────────────

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = Swish()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = rearrange(h, 'b c h w -> b (h w) c')
        h, _ = self.attn(h, h, h)
        h = rearrange(h, 'b (h w) c -> b c h w', h=H, w=W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    ADM-style UNet.
    forward(x, t, return_features=True) 시 (noise_pred, bottleneck_feature) 반환.
    논문: 특징 추출은 bottleneck layer에서 수행 (Section 3.3.2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: list = None,
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 2, 2),
        num_heads: int = 4,
    ):
        super().__init__()
        if attention_resolutions is None:
            attention_resolutions = [16, 8]

        self.model_channels = model_channels
        time_dim = model_channels * 4

        self.time_emb = TimeEmbedding(model_channels)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = model_channels
        input_ch_list = [ch]
        cur_res = 32  # 기본 해상도 (CIFAR-10), ImageNet은 config에서 조정

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout)])
                if cur_res in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                self.down_blocks.append(block)
                ch = out_ch
                input_ch_list.append(ch)
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                input_ch_list.append(ch)
                cur_res //= 2

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid_res1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_res2 = ResBlock(ch, ch, time_dim, dropout)
        self.bottleneck_channels = ch  # 특징 차원 노출

        # ── Decoder ─────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = input_ch_list.pop()
                block = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, time_dim, dropout)])
                if cur_res in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                self.up_blocks.append(block)
                ch = out_ch
            if level != 0:
                self.up_samples.append(Upsample(ch))
                cur_res *= 2

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        self.act = Swish()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, return_features: bool = False
    ):
        """
        Args:
            x: noisy image [B, C, H, W]
            t: timestep [B]
            return_features: True이면 (noise_pred, bottleneck_feat) 반환

        Returns:
            noise_pred: [B, C, H, W]
            bottleneck_feat (optional): [B, C_mid, H', W'] → GAP 후 [B, C_mid]
        """
        t_emb = self.time_emb(t)

        # Encoder
        h = self.input_conv(x)
        skips = [h]
        ds_idx = 0

        for block_group in self.down_blocks:
            for layer in block_group:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)
            # 마지막 레벨 제외 다운샘플
            if ds_idx < len(self.down_samples) and len(skips) % (len(self.down_blocks) // len(self.down_samples) + 1) == 0:
                pass  # 단순화된 다운샘플 처리

        # 다운샘플 별도 처리
        h = self.input_conv(x)
        skips = [h]
        ds_idx = 0
        block_idx = 0
        num_levels = len(self.down_blocks)

        # 인코더 재구성 (스킵 연결 올바르게)
        h = self.input_conv(x)
        skips = [h]

        block_iter = iter(self.down_blocks)
        ds_iter = iter(self.down_samples)
        blocks_per_level = [len(self.down_blocks) // len(set([1,2,2,2]))]

        # 단순하고 명확한 forward
        h = self.input_conv(x)
        skips = [h]
        block_list = list(self.down_blocks)
        ds_list = list(self.down_samples)
        b_idx, d_idx = 0, 0

        num_res = 2  # num_res_blocks
        for level in range(len(ds_list) + 1):
            for _ in range(num_res):
                if b_idx < len(block_list):
                    for layer in block_list[b_idx]:
                        h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
                    skips.append(h)
                    b_idx += 1
            if d_idx < len(ds_list):
                h = ds_list[d_idx](h)
                skips.append(h)
                d_idx += 1

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Global Average Pool → bottleneck feature vector
        bottleneck_feat = h.mean(dim=[2, 3])  # [B, C_mid]

        # Decoder
        up_list = list(self.up_blocks)
        ups_list = list(self.up_samples)
        u_idx, us_idx = 0, 0

        for level in reversed(range(len(ups_list) + 1)):
            for i in range(num_res + 1):
                if u_idx < len(up_list) and skips:
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    for layer in up_list[u_idx]:
                        h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
                    u_idx += 1
            if us_idx < len(ups_list):
                h = ups_list[us_idx](h)
                us_idx += 1

        h = self.act(self.out_norm(h))
        noise_pred = self.out_conv(h)

        if return_features:
            return noise_pred, bottleneck_feat
        return noise_pred
