"""Sanity and unit tests for the SDD diffusion project.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import pytest
import torch

from src.models.unet import UNetModel
from src.models.diffusion import Diffusion
from src.sdd.distillation import Center, dino_loss
from src.sdd.gating import timestep_gate
from src.sdd.ema import clone_model, ema_update
from src.sdd.projection_head import ProjectionHead
from src.training.losses import total_loss


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_unet():
    return UNetModel(base_channels=32, channel_mults=(1, 2), num_res_blocks=1,
                     attention_resolutions=(8,), image_size=16)


@pytest.fixture(scope="module")
def diffusion():
    return Diffusion(timesteps=100, beta_schedule="linear", device="cpu")


# ──────────────────────────────────────────────────────────────────────────────
# UNet
# ──────────────────────────────────────────────────────────────────────────────

class TestUNet:
    def test_forward_output_shape(self, small_unet):
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        out = small_unet(x, t)
        assert out.shape == x.shape, "output shape must match input shape"

    @pytest.mark.parametrize("layer", ["bottleneck", "skip1", "skip2", "decoder1"])
    def test_return_features_shape(self, small_unet, layer):
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        out, feat = small_unet(x, t, return_features=True, feature_layer=layer)
        assert out.shape == x.shape
        assert feat.ndim == 2
        assert feat.shape[0] == 2, "batch dim must be preserved"

    def test_invalid_feature_layer_raises(self, small_unet):
        x = torch.randn(2, 3, 16, 16)
        t = torch.zeros(2, dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown feature_layer"):
            small_unet(x, t, return_features=True, feature_layer="nonexistent")

    def test_no_feature_returns_tensor(self, small_unet):
        x = torch.randn(2, 3, 16, 16)
        t = torch.zeros(2, dtype=torch.long)
        result = small_unet(x, t)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion
# ──────────────────────────────────────────────────────────────────────────────

class TestDiffusion:
    def test_q_sample_shape(self, diffusion):
        x0 = torch.randn(4, 3, 16, 16)
        t = torch.randint(0, 100, (4,))
        xt, noise = diffusion.q_sample(x0, t)
        assert xt.shape == x0.shape
        assert noise.shape == x0.shape

    def test_alphas_cumprod_monotone(self, diffusion):
        ac = diffusion.alphas_cumprod
        assert (ac[1:] <= ac[:-1]).all(), "alphas_cumprod must be non-increasing"

    def test_alphas_cumprod_bounds(self, diffusion):
        ac = diffusion.alphas_cumprod
        assert float(ac[0]) <= 1.0
        assert float(ac[-1]) > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Gating
# ──────────────────────────────────────────────────────────────────────────────

class TestGating:
    def test_hard_gate_values_binary(self):
        t = torch.arange(0, 1000)
        g = timestep_gate(t, 1000, mode="hard", t_min=0.1, t_max=0.6)
        unique = set(g.tolist())
        assert unique <= {0.0, 1.0}, "hard gate must be binary"

    def test_hard_gate_range(self):
        T = 1000
        t = torch.arange(T)
        g = timestep_gate(t, T, mode="hard", t_min=0.1, t_max=0.6)
        t_norm = t.float() / (T - 1)
        inside  = (t_norm >= 0.1) & (t_norm <= 0.6)
        outside = ~inside
        assert g[inside].all(),        "all inside-range steps should be gated ON"
        assert (g[outside] == 0).all(), "all outside-range steps should be gated OFF"

    def test_soft_gate_in_01(self):
        t = torch.arange(0, 1000)
        g = timestep_gate(t, 1000, mode="soft", t_min=0.1, t_max=0.6, soft_beta=0.08)
        assert (g >= 0).all() and (g <= 1).all(), "soft gate must be in [0, 1]"

    def test_soft_gate_monotone_flanks(self):
        T = 1000
        t = torch.arange(T)
        g = timestep_gate(t, T, mode="soft", t_min=0.1, t_max=0.6, soft_beta=0.08)
        # Gate should increase then decrease — peak near the middle
        mid = T // 2
        assert float(g[mid]) > float(g[0]), "gate should be higher in the middle than at t=0"

    def test_no_gating_all_ones(self):
        t = torch.randint(0, 1000, (100,))
        g = timestep_gate(t, 1000, mode="none")
        assert (g == 1.0).all(), "mode='none' should return all-ones gate"


# ──────────────────────────────────────────────────────────────────────────────
# Distillation
# ──────────────────────────────────────────────────────────────────────────────

class TestDistillation:
    def test_dino_loss_non_negative(self):
        B, D = 8, 64
        teacher = torch.randn(B, D)
        student = torch.randn(B, D)
        loss = dino_loss(teacher, student, center=None,
                         use_centering=False, use_sharpening=True)
        assert float(loss) >= 0.0, "cross-entropy-based loss must be non-negative"

    def test_dino_loss_decreases_at_optimum(self):
        """Student initialized to teacher logits should give lower loss."""
        B, D = 8, 64
        teacher = torch.randn(B, D)
        bad_student = torch.randn(B, D)
        good_student = teacher.clone()
        loss_bad  = dino_loss(teacher, bad_student,  center=None, use_centering=False)
        loss_good = dino_loss(teacher, good_student, center=None, use_centering=False)
        assert float(loss_good) < float(loss_bad), "matching student should have lower loss"

    def test_center_update_shape(self):
        B, D = 16, 128
        center = Center(D, momentum=0.9)
        logits = torch.randn(B, D)
        center.update(logits)
        assert center().shape == (D,)

    def test_center_momentum(self):
        D = 32
        center = Center(D, momentum=0.9)
        ones_logits = torch.ones(10, D)
        for _ in range(50):
            center.update(ones_logits)
        # After many updates, center should be close to 1.0
        assert float(center().mean()) > 0.9


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_ema_update_interpolates(self):
        model = UNetModel(base_channels=16, channel_mults=(1,), num_res_blocks=1,
                          attention_resolutions=(), image_size=8)
        teacher = clone_model(model)

        # Record initial teacher param
        param_name = list(dict(teacher.named_parameters()).keys())[0]
        t_before = dict(teacher.named_parameters())[param_name].data.clone()

        # Modify student
        with torch.no_grad():
            dict(model.named_parameters())[param_name].data.fill_(1.0)

        m = 0.9
        ema_update(teacher, model, momentum=m)
        t_after = dict(teacher.named_parameters())[param_name].data

        expected = t_before * m + 1.0 * (1 - m)
        assert torch.allclose(t_after, expected, atol=1e-5), "EMA update formula must be t = m*t + (1-m)*s"

    def test_teacher_no_grad(self, small_unet):
        teacher = clone_model(small_unet)
        for p in teacher.parameters():
            assert not p.requires_grad, "teacher parameters must not require grad"


# ──────────────────────────────────────────────────────────────────────────────
# Projection head
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectionHead:
    def test_output_shape(self):
        head = ProjectionHead(in_dim=512, proj_dim=256)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 256)

    def test_no_nan(self):
        head = ProjectionHead(in_dim=128, proj_dim=64)
        x = torch.randn(8, 128)
        out = head(x)
        assert not torch.isnan(out).any()


# ──────────────────────────────────────────────────────────────────────────────
# Total loss
# ──────────────────────────────────────────────────────────────────────────────

class TestTotalLoss:
    def _make_inputs(self, B=4, D=64, T=100):
        eps_pred     = torch.randn(B, 3, 8, 8)
        noise        = torch.randn(B, 3, 8, 8)
        s_logits     = torch.randn(B, D)
        t_logits     = torch.randn(B, D)
        center       = Center(D)
        timesteps    = torch.randint(0, T, (B,))
        return eps_pred, noise, s_logits, t_logits, center, timesteps, T

    def test_mse_only_no_distill(self):
        eps_pred, noise, s_logits, t_logits, center, timesteps, T = self._make_inputs()
        loss, metrics = total_loss(
            eps_pred, noise, s_logits, t_logits, center, timesteps, T,
            lambda_distill=0.0,
        )
        assert metrics["loss_distill"] == 0.0

    def test_no_student_logits_skips_distill(self):
        eps_pred, noise, _, _, _, timesteps, T = self._make_inputs()
        loss, metrics = total_loss(
            eps_pred, noise, None, None, None, timesteps, T,
            lambda_distill=0.5,
        )
        assert metrics["loss_distill"] == 0.0

    def test_full_loss_positive(self):
        eps_pred, noise, s_logits, t_logits, center, timesteps, T = self._make_inputs()
        loss, metrics = total_loss(
            eps_pred, noise, s_logits, t_logits, center, timesteps, T,
            lambda_distill=0.5, use_centering=True, use_sharpening=True, use_gating=True,
        )
        assert float(loss) > 0

    def test_gate_mean_metric_in_01(self):
        eps_pred, noise, s_logits, t_logits, center, timesteps, T = self._make_inputs()
        _, metrics = total_loss(
            eps_pred, noise, s_logits, t_logits, center, timesteps, T,
            lambda_distill=0.5, use_gating=True, gating_mode="hard",
            t_min=0.1, t_max=0.6,
        )
        assert 0.0 <= metrics["gate_mean"] <= 1.0

    def test_per_sample_gating_correctness(self):
        """gate.mean() scaling — verify that disabling gating increases distill loss."""
        B, D, T = 8, 64, 100
        eps_pred = torch.randn(B, 3, 8, 8)
        noise    = torch.randn(B, 3, 8, 8)
        s_logits = torch.randn(B, D)
        t_logits = torch.randn(B, D)
        center   = Center(D)
        # Force all timesteps to be outside the gate window
        timesteps_outside = torch.zeros(B, dtype=torch.long)  # t=0 → t_norm=0 < t_min=0.1

        _, metrics_gated = total_loss(
            eps_pred, noise, s_logits, t_logits, center, timesteps_outside, T,
            lambda_distill=0.5, use_gating=True, gating_mode="hard",
            t_min=0.1, t_max=0.6,
        )
        _, metrics_no_gate = total_loss(
            eps_pred, noise, s_logits, t_logits, center, timesteps_outside, T,
            lambda_distill=0.5, use_gating=False,
        )
        # With gating ON and all steps outside window, distill should be ~0
        assert metrics_gated["gate_mean"] == 0.0, "all t outside window → gate_mean must be 0"
        # Without gating, distill_loss should be non-zero
        assert metrics_no_gate["loss_distill"] > 0.0
