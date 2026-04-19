from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = self.proj(attn_out)
        return attn_out.transpose(1, 2).view(b, c, h, w) + x_in


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        self.image_size = image_size
        self.base_channels = base_channels
        self.time_dim = base_channels * 4
        self.attention_resolutions = set(attention_resolutions)

        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels = [ch]
        cur_res = image_size

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, self.time_dim, dropout=dropout))
                ch = out_ch
                self.skip_channels.append(ch)
                if cur_res in self.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch))
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                cur_res //= 2
                self.skip_channels.append(ch)

        self.mid_block1 = ResBlock(ch, ch, self.time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, self.time_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        up_mults = list(channel_mults)[::-1]
        up_res = cur_res
        for i, mult in enumerate(up_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop() if self.skip_channels else out_ch
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, self.time_dim, dropout=dropout))
                ch = out_ch
                if up_res in self.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch))
            if i != len(up_mults) - 1:
                self.upsamples.append(Upsample(ch))
                up_res *= 2

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_features: bool = False):
        t_emb = self.time_embed(sinusoidal_embedding(t, self.base_channels))
        h = self.input_conv(x)
        skips = [h]

        current = h
        downsample_idx = 0
        for module in self.down_blocks:
            if isinstance(module, ResBlock):
                current = module(current, t_emb)
                skips.append(current)
            elif isinstance(module, AttentionBlock):
                current = module(current)

        for ds in self.downsamples:
            current = ds(current)
            skips.append(current)

        bottleneck = self.mid_block1(current, t_emb)
        bottleneck = self.mid_attn(bottleneck)
        bottleneck = self.mid_block2(bottleneck, t_emb)

        current = bottleneck
        for module in self.up_blocks:
            if isinstance(module, ResBlock):
                if skips:
                    skip = skips.pop()
                    if skip.shape[2:] != current.shape[2:]:
                        skip = F.interpolate(skip, size=current.shape[2:], mode="nearest")
                    current = torch.cat([current, skip], dim=1)
                current = module(current, t_emb)
            elif isinstance(module, AttentionBlock):
                current = module(current)

        for us in self.upsamples:
            current = us(current)

        out = self.out_conv(F.silu(self.out_norm(current)))
        if return_features:
            feat = bottleneck.mean(dim=(2, 3))
            return out, feat
        return out
