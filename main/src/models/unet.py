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
    """UNet diffusion backbone with multi-layer feature extraction support.

    The encoder and decoder are stored as level-structured lists so that
    Downsample / Upsample modules are applied at the correct channel width
    (end of each level, before the next level's ResBlocks increase channels).

    down_levels: list[list[nn.Module]]  — one inner list per encoder level
    up_levels:   list[list[nn.Module]]  — one inner list per decoder level
    downsamples / upsamples: one module per level boundary
    """

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
        self._channel_mults = channel_mults
        self.time_dim = base_channels * 4
        self.attention_resolutions = set(attention_resolutions)

        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each level = list of ResBlock/AttentionBlock modules.
        # Downsample sits between levels (not inside a level).
        self.down_levels: nn.ModuleList = nn.ModuleList()   # each element is nn.ModuleList
        self.downsamples: nn.ModuleList = nn.ModuleList()

        ch = base_channels
        cur_res = image_size
        skip_channels: list[int] = [ch]          # track for decoder skip connection dims

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks: list[nn.Module] = []
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, self.time_dim, dropout=dropout))
                ch = out_ch
                skip_channels.append(ch)
                if cur_res in self.attention_resolutions:
                    level_blocks.append(AttentionBlock(ch))
            self.down_levels.append(nn.ModuleList(level_blocks))
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                cur_res //= 2
                skip_channels.append(ch)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_block1 = ResBlock(ch, ch, self.time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, self.time_dim, dropout=dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up_levels: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()

        up_mults = list(channel_mults)[::-1]
        up_res = cur_res
        for i, mult in enumerate(up_mults):
            out_ch = base_channels * mult
            level_blocks = []
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop() if skip_channels else out_ch
                level_blocks.append(ResBlock(ch + skip_ch, out_ch, self.time_dim, dropout=dropout))
                ch = out_ch
                if up_res in self.attention_resolutions:
                    level_blocks.append(AttentionBlock(ch))
            self.up_levels.append(nn.ModuleList(level_blocks))
            if i != len(up_mults) - 1:
                self.upsamples.append(Upsample(ch))
                up_res *= 2

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    # ── Compatibility shims for old code that used down_blocks / up_blocks ──

    @property
    def down_blocks(self) -> list[nn.Module]:
        """Flat list of all encoder-level modules (ResBlocks + AttentionBlocks)."""
        return [m for level in self.down_levels for m in level]

    @property
    def up_blocks(self) -> list[nn.Module]:
        """Flat list of all decoder-level modules."""
        return [m for level in self.up_levels for m in level]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
        feature_layer: str = "bottleneck",
    ):
        """
        Args:
            x: noisy image tensor (B, C, H, W)
            t: timestep tensor (B,)
            return_features: if True, also return a feature vector (B, D)
            feature_layer: one of:
                "bottleneck" — global avg-pool of mid-block output  [default]
                "skip1"      — last encoder ResBlock output before bottleneck
                "skip2"      — second-to-last encoder ResBlock output
                "decoder1"   — first decoder ResBlock output
        """
        _valid = {"bottleneck", "skip1", "skip2", "decoder1"}
        if return_features and feature_layer not in _valid:
            raise ValueError(
                f"Unknown feature_layer: {feature_layer!r}. "
                f"Choose from {sorted(_valid)}"
            )

        t_emb = self.time_embed(sinusoidal_embedding(t, self.base_channels))
        h = self.input_conv(x)
        skips: list[torch.Tensor] = [h]
        encoder_res_outs: list[torch.Tensor] = []

        current = h

        # ── Encoder: iterate levels; Downsample between levels ───────────────
        for level_idx, level in enumerate(self.down_levels):
            for module in level:
                if isinstance(module, ResBlock):
                    current = module(current, t_emb)
                    skips.append(current)
                    encoder_res_outs.append(current)
                elif isinstance(module, AttentionBlock):
                    current = module(current)
            if level_idx < len(self.downsamples):
                current = self.downsamples[level_idx](current)
                skips.append(current)

        # ── Bottleneck ────────────────────────────────────────────────────────
        bottleneck = self.mid_block1(current, t_emb)
        bottleneck = self.mid_attn(bottleneck)
        bottleneck = self.mid_block2(bottleneck, t_emb)

        current = bottleneck
        decoder_res_outs: list[torch.Tensor] = []

        # ── Decoder: iterate levels; Upsample between levels ─────────────────
        for level_idx, level in enumerate(self.up_levels):
            for module in level:
                if isinstance(module, ResBlock):
                    if skips:
                        skip = skips.pop()
                        if skip.shape[2:] != current.shape[2:]:
                            skip = F.interpolate(skip, size=current.shape[2:], mode="nearest")
                        current = torch.cat([current, skip], dim=1)
                    current = module(current, t_emb)
                    decoder_res_outs.append(current)
                elif isinstance(module, AttentionBlock):
                    current = module(current)
            if level_idx < len(self.upsamples):
                current = self.upsamples[level_idx](current)

        out = self.out_conv(F.silu(self.out_norm(current)))

        if return_features:
            if feature_layer == "bottleneck":
                feat = bottleneck.mean(dim=(2, 3))
            elif feature_layer == "skip1":
                src = encoder_res_outs[-1] if encoder_res_outs else bottleneck
                feat = src.mean(dim=(2, 3))
            elif feature_layer == "skip2":
                src = (encoder_res_outs[-2] if len(encoder_res_outs) >= 2
                       else (encoder_res_outs[-1] if encoder_res_outs else bottleneck))
                feat = src.mean(dim=(2, 3))
            elif feature_layer == "decoder1":
                src = decoder_res_outs[0] if decoder_res_outs else bottleneck
                feat = src.mean(dim=(2, 3))
            return out, feat
        return out
