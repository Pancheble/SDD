from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.evaluation.feature_extract import extract_features
from src.evaluation.fid import FIDEvaluator
from src.evaluation.linear_probe import train_linear_probe
from src.models.unet import UNetModel
from src.training.trainer import SDDTrainer
from src.utils.seed import set_seed


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_cfg(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def save_cfg(cfg: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return path


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(ops)


def build_datasets(cfg: Dict[str, Any]):
    image_size = cfg["dataset"]["image_size"]
    name = cfg["dataset"]["name"].lower()
    train_tfm = build_transforms(image_size, train=True)
    eval_tfm = build_transforms(image_size, train=False)

    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=cfg["dataset"]["root"], train=True, download=True, transform=train_tfm)
        test_ds = datasets.CIFAR10(root=cfg["dataset"]["root"], train=False, download=True, transform=eval_tfm)
    elif name == "tiny_imagenet":
        train_dir = Path(cfg["dataset"].get("train_dir", "./data/tiny-imagenet-200/train"))
        val_dir = Path(cfg["dataset"].get("val_dir", "./data/tiny-imagenet-200/val"))
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                "Tiny ImageNet path not found. Expected train_dir and val_dir to point to the extracted dataset."
            )
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfm)
        test_ds = datasets.ImageFolder(root=str(val_dir), transform=eval_tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_ds, test_ds


def build_loaders(cfg: Dict[str, Any]):
    train_ds, test_ds = build_datasets(cfg)
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 4)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, test_loader


def build_model(cfg: Dict[str, Any]) -> UNetModel:
    return UNetModel(
        base_channels=cfg["model"]["channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
        dropout=cfg["model"]["dropout"],
        image_size=cfg["dataset"]["image_size"],
    )


def make_optimizer(trainer: SDDTrainer, cfg: Dict[str, Any]):
    params = list(trainer.model.parameters())
    if trainer.proj_student is not None:
        params += list(trainer.proj_student.parameters())
    return torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


def maybe_init_wandb(cfg: Dict[str, Any], name: str | None = None):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    import wandb

    return wandb.init(
        project=wandb_cfg.get("project", cfg.get("project", "sdd_diffusion")),
        entity=wandb_cfg.get("entity"),
        name=name or cfg.get("run_name"),
        tags=wandb_cfg.get("tags", []),
        config=cfg,
        reinit=True,
    )


def build_trainer(cfg: Dict[str, Any], device: str | torch.device | None = None) -> SDDTrainer:
    return SDDTrainer(build_model(cfg), cfg, device or DEFAULT_DEVICE)




def make_checkpoint_path(cfg: Dict[str, Any], suffix: str = "last") -> Path:
    out_dir = Path(cfg["output"]["dir"]) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{cfg['run_name']}_{suffix}.pt"


def save_checkpoint(trainer: SDDTrainer, optimizer, cfg: Dict[str, Any], epoch: int, path: str | Path | None = None) -> Path:
    path = Path(path) if path is not None else make_checkpoint_path(cfg, "last")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": trainer.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "state": trainer.state.__dict__,
        "teacher": trainer.teacher.state_dict() if trainer.teacher is not None else None,
        "proj_student": trainer.proj_student.state_dict() if trainer.proj_student is not None else None,
        "proj_teacher": trainer.proj_teacher.state_dict() if trainer.proj_teacher is not None else None,
        "center": trainer.center.value if trainer.center is not None else None,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(trainer: SDDTrainer, optimizer=None, path: str | Path | None = None):
    if path is None:
        raise ValueError("path is required to load a checkpoint")
    ckpt = torch.load(path, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model"])
    if ckpt.get("teacher") is not None and trainer.teacher is not None:
        trainer.teacher.load_state_dict(ckpt["teacher"])
    if ckpt.get("proj_student") is not None and trainer.proj_student is not None:
        trainer.proj_student.load_state_dict(ckpt["proj_student"])
    if ckpt.get("proj_teacher") is not None and trainer.proj_teacher is not None:
        trainer.proj_teacher.load_state_dict(ckpt["proj_teacher"])
    if ckpt.get("center") is not None and trainer.center is not None:
        trainer.center.value.copy_(ckpt["center"].to(trainer.device))
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "state" in ckpt:
        trainer.state.step = ckpt["state"].get("step", 0)
        trainer.state.epoch = ckpt["state"].get("epoch", 0)
    return ckpt

def train_epochs(
    trainer: SDDTrainer,
    train_loader: DataLoader,
    cfg: Dict[str, Any],
    optimizer,
    run=None,
    val_loader: DataLoader | None = None,
) -> pd.DataFrame:
    use_amp = cfg["train"]["mixed_precision"] and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rows = []

    save_every = cfg["train"].get("save_every", 1)
    last_ckpt = None

    for epoch in range(cfg["train"]["epochs"]):
        trainer.state.epoch = epoch
        metrics = trainer.train_one_epoch(train_loader, optimizer, scaler=scaler, wandb_run=run)
        row = {"epoch": epoch, **metrics}
        rows.append(row)

        if run is not None:
            run.log({"epoch": epoch, **metrics})

        if val_loader is not None and (epoch + 1) % cfg["train"].get("eval_every", 1) == 0:
            val_metrics = evaluate_generation(trainer, val_loader, cfg)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if run is not None:
                run.log({**{f"val_{k}": v for k, v in val_metrics.items()}, "epoch": epoch})

        if (epoch + 1) % save_every == 0 or epoch == cfg["train"]["epochs"] - 1:
            last_ckpt = save_checkpoint(trainer, optimizer, cfg, epoch)

    history = pd.DataFrame(rows)
    out_dir = Path(cfg["output"]["dir"]) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / f"{cfg['run_name']}_history.csv", index=False)
    if last_ckpt is not None:
        history.attrs["checkpoint_path"] = str(last_ckpt)
    return history


@torch.no_grad()
def evaluate_generation(trainer: SDDTrainer, loader: DataLoader, cfg: Dict[str, Any]) -> Dict[str, float]:
    trainer.model.eval()
    device = trainer.device
    fid = FIDEvaluator(device=device)

    target_samples = cfg["train"].get("fid_num_samples", 512)
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        x = (x * 0.5 + 0.5).clamp(0, 1)
        fid.update_real((x * 255).to(torch.uint8))
        seen += x.size(0)
        if seen >= target_samples:
            break

    sample_shape = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
    remaining = target_samples
    while remaining > 0:
        n = min(remaining, cfg["train"].get("num_samples_preview", 64))
        fake = trainer.sample(n=n, shape=sample_shape)
        fake = (fake * 0.5 + 0.5).clamp(0, 1)
        fid.update_fake((fake * 255).to(torch.uint8))
        remaining -= n
    return {"fid": fid.compute()}


@torch.no_grad()
def collect_features(trainer: SDDTrainer, loader: DataLoader):
    return extract_features(trainer.model, loader, trainer.device)


def run_linear_probe(trainer: SDDTrainer, train_loader: DataLoader, test_loader: DataLoader, cfg: Dict[str, Any]):
    train_feats, train_labels = collect_features(trainer, train_loader)
    test_feats, test_labels = collect_features(trainer, test_loader)
    _, acc = train_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=test_feats,
        val_labels=test_labels,
        num_classes=cfg["dataset"]["num_classes"],
        epochs=cfg.get("probe", {}).get("epochs", 50),
        lr=cfg.get("probe", {}).get("lr", 1e-3),
        device=trainer.device,
    )
    return acc




def build_trainer_from_checkpoint(cfg: Dict[str, Any], checkpoint_path: str | Path, device: str | None = None) -> SDDTrainer:
    device = device or DEFAULT_DEVICE
    trainer = build_trainer(cfg, device=device)
    load_checkpoint(trainer, path=checkpoint_path)
    return trainer

def run_experiment(cfg: Dict[str, Any], device: str | None = None, with_eval: bool = False):
    device = device or DEFAULT_DEVICE
    set_seed(cfg["train"]["seed"])
    train_loader, test_loader = build_loaders(cfg)
    trainer = build_trainer(cfg, device=device)
    optimizer = make_optimizer(trainer, cfg)
    run = maybe_init_wandb(cfg)
    history = train_epochs(trainer, train_loader, cfg, optimizer, run=run, val_loader=test_loader if with_eval else None)
    results = {"history": history, "trainer": trainer, "train_loader": train_loader, "test_loader": test_loader}
    if with_eval:
        results["fid"] = evaluate_generation(trainer, test_loader, cfg)["fid"]
        results["linear_probe_acc"] = run_linear_probe(trainer, train_loader, test_loader, cfg)
    if run is not None:
        run.finish()
    return results




def load_and_evaluate(cfg: Dict[str, Any], checkpoint_path: str | Path, device: str | None = None):
    trainer = build_trainer_from_checkpoint(cfg, checkpoint_path, device=device)
    train_loader, test_loader = build_loaders(cfg)
    return {
        "fid": evaluate_generation(trainer, test_loader, cfg)["fid"],
        "linear_probe_acc": run_linear_probe(trainer, train_loader, test_loader, cfg),
    }

def run_ablation_suite(base_cfg: Dict[str, Any], variants: Dict[str, Dict[str, Any]], device: str | None = None):
    outputs = []
    for name, overrides in variants.items():
        cfg = deep_update(base_cfg, overrides)
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_{name}"
        save_cfg(cfg, Path(base_cfg["output"]["dir"]) / "configs" / f"{cfg['run_name']}.yaml")
        result = run_experiment(cfg, device=device, with_eval=True)
        outputs.append({"variant": name, "fid": result.get("fid"), "linear_probe_acc": result.get("linear_probe_acc")})
    return pd.DataFrame(outputs)


def run_timestep_sweep(base_cfg: Dict[str, Any], sweep: Iterable[tuple[float, float]], device: str | None = None):
    rows = []
    for t_min, t_max in sweep:
        cfg = deep_update(base_cfg, {"sdd": {"gating": {"t_min": t_min, "t_max": t_max}}})
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_gate_{t_min:.2f}_{t_max:.2f}"
        result = run_experiment(cfg, device=device, with_eval=True)
        rows.append({"t_min": t_min, "t_max": t_max, "fid": result.get("fid"), "linear_probe_acc": result.get("linear_probe_acc")})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Feature-layer ablation
# Compare bottleneck / skip1 / skip2 / decoder1 as distillation targets
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_LAYERS = ["bottleneck", "skip1", "skip2", "decoder1"]


def run_feature_layer_ablation(
    base_cfg: Dict[str, Any],
    layers: list[str] | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Train one SDD run per feature layer and compare FID + linear probe.

    Args:
        base_cfg: base experiment config (should have sdd.enabled=True)
        layers: list of layer names to compare; defaults to all four
        device: compute device

    Returns:
        DataFrame with columns [feature_layer, fid, linear_probe_acc]
    """
    layers = layers or FEATURE_LAYERS
    rows = []
    for layer in layers:
        cfg = deep_update(base_cfg, {"sdd": {"feature_layer": layer}})
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_feat_{layer}"
        save_cfg(cfg, Path(base_cfg["output"]["dir"]) / "configs" / f"{cfg['run_name']}.yaml")
        result = run_experiment(cfg, device=device, with_eval=True)
        rows.append(
            {
                "feature_layer": layer,
                "fid": result.get("fid"),
                "linear_probe_acc": result.get("linear_probe_acc"),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Training-curve logger
# Collect per-epoch FID + linear probe acc to plot convergence
# ─────────────────────────────────────────────────────────────────────────────

def train_with_curves(
    cfg: Dict[str, Any],
    device: str | None = None,
    eval_every: int | None = None,
) -> pd.DataFrame:
    """Train and record FID + linear_probe_acc every N epochs.

    Args:
        cfg: experiment config
        device: compute device
        eval_every: evaluate every this many epochs; defaults to cfg eval_every

    Returns:
        DataFrame with columns [epoch, loss_total, loss_mse, loss_distill,
                                 gate_mean, fid, linear_probe_acc]
    """
    device = device or DEFAULT_DEVICE
    set_seed(cfg["train"]["seed"])
    train_loader, test_loader = build_loaders(cfg)
    trainer = build_trainer(cfg, device=device)
    optimizer = make_optimizer(trainer, cfg)
    run = maybe_init_wandb(cfg)
    eval_every = eval_every or cfg["train"].get("eval_every", 10)

    use_amp = cfg["train"]["mixed_precision"] and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        trainer.state.epoch = epoch
        metrics = trainer.train_one_epoch(train_loader, optimizer, scaler=scaler, wandb_run=run)
        row: Dict[str, Any] = {"epoch": epoch, **metrics}

        if (epoch + 1) % eval_every == 0 or epoch == cfg["train"]["epochs"] - 1:
            gen_metrics = evaluate_generation(trainer, test_loader, cfg)
            probe_acc = run_linear_probe(trainer, train_loader, test_loader, cfg)
            row["fid"] = gen_metrics["fid"]
            row["linear_probe_acc"] = probe_acc
            if run is not None:
                run.log({"epoch": epoch, "fid": gen_metrics["fid"], "linear_probe_acc": probe_acc})

        rows.append(row)

        if (epoch + 1) % cfg["train"].get("save_every", 10) == 0:
            save_checkpoint(trainer, optimizer, cfg, epoch)

    if run is not None:
        run.finish()

    history = pd.DataFrame(rows)
    out_dir = Path(cfg["output"]["dir"]) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / f"{cfg['run_name']}_curve.csv", index=False)
    return history


def compare_training_curves(
    base_cfg: Dict[str, Any],
    variants: Dict[str, Dict[str, Any]],
    device: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run train_with_curves for each variant and return {name: DataFrame}."""
    results = {}
    for name, overrides in variants.items():
        cfg = deep_update(base_cfg, overrides)
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_{name}_curve"
        results[name] = train_with_curves(cfg, device=device)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Gating distribution analysis
# Record per-step timestep histogram and gate_mean to understand which
# timesteps actually receive distillation signal
# ─────────────────────────────────────────────────────────────────────────────

def collect_gate_histogram(
    trainer,
    loader: DataLoader,
    n_batches: int = 50,
) -> Dict[str, Any]:
    """Collect timestep samples and their gate values for one epoch slice.

    Returns dict with:
        timesteps: np.ndarray shape (N,)
        gate_values: np.ndarray shape (N,)
        t_min, t_max, mode: gating config used
    """
    import numpy as np
    from src.sdd.gating import timestep_gate

    cfg_gate = trainer.cfg["sdd"]["gating"]
    t_min = cfg_gate["t_min"]
    t_max = cfg_gate["t_max"]
    mode = cfg_gate["mode"]
    soft_beta = cfg_gate.get("soft_beta", 0.08)
    T = trainer.diffusion.timesteps

    all_t, all_g = [], []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(trainer.device)
        t = torch.randint(0, T, (x.size(0),), device=trainer.device)
        g = timestep_gate(t, T, mode=mode, t_min=t_min, t_max=t_max, soft_beta=soft_beta)
        all_t.append(t.cpu().numpy())
        all_g.append(g.cpu().numpy())

    return {
        "timesteps": np.concatenate(all_t),
        "gate_values": np.concatenate(all_g),
        "t_min": t_min,
        "t_max": t_max,
        "mode": mode,
        "T": T,
    }


def run_gating_analysis(
    base_cfg: Dict[str, Any],
    gating_configs: Dict[str, Dict[str, Any]] | None = None,
    device: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare gate distributions for hard vs soft gating (and custom configs).

    Args:
        base_cfg: full experiment config
        gating_configs: optional dict of {name: sdd.gating overrides};
                        defaults to hard vs soft comparison
        device: compute device

    Returns:
        dict {name: gate_histogram_dict}
    """
    if gating_configs is None:
        gating_configs = {
            "hard": {"sdd": {"gating": {"mode": "hard"}}},
            "soft": {"sdd": {"gating": {"mode": "soft", "soft_beta": 0.08}}},
            "soft_narrow": {"sdd": {"gating": {"mode": "soft", "soft_beta": 0.04}}},
        }

    device = device or DEFAULT_DEVICE
    _, train_loader = build_loaders(base_cfg)  # we only need the timestep sampler
    train_loader, _ = build_loaders(base_cfg)

    results = {}
    for name, overrides in gating_configs.items():
        cfg = deep_update(base_cfg, overrides)
        # We only need the trainer to get gating config; no full training
        trainer = build_trainer(cfg, device=device)
        results[name] = collect_gate_histogram(trainer, train_loader)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: EMA momentum sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_ema_momentum_sweep(
    base_cfg: Dict[str, Any],
    momentum_values: list[float] | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Train with different EMA teacher momentum values and compare metrics.

    Args:
        base_cfg: base experiment config
        momentum_values: list of momentum floats; defaults to [0.990, 0.996, 0.999]
        device: compute device

    Returns:
        DataFrame with columns [teacher_momentum, fid, linear_probe_acc]
    """
    momentum_values = momentum_values or [0.990, 0.996, 0.999]
    rows = []
    for m in momentum_values:
        cfg = deep_update(base_cfg, {"sdd": {"teacher_momentum": m}})
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_ema_{str(m).replace('.', '')}"
        save_cfg(cfg, Path(base_cfg["output"]["dir"]) / "configs" / f"{cfg['run_name']}.yaml")
        result = run_experiment(cfg, device=device, with_eval=True)
        rows.append(
            {
                "teacher_momentum": m,
                "fid": result.get("fid"),
                "linear_probe_acc": result.get("linear_probe_acc"),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Generated sample grid
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample_grid(
    trainer,
    cfg: Dict[str, Any],
    n: int = 64,
    seed: int = 0,
    save_path: str | Path | None = None,
):
    """Generate n samples and return them as a (n, C, H, W) uint8 tensor.

    Also saves a PNG grid to save_path (or outputs/samples/<run_name>_grid.png).

    Args:
        trainer: trained SDDTrainer
        cfg: experiment config
        n: number of samples to generate
        seed: random seed for reproducibility
        save_path: override output file path

    Returns:
        samples tensor, shape (n, C, H, W) in [0, 255] uint8
    """
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    shape = (
        3,
        cfg["dataset"]["image_size"],
        cfg["dataset"]["image_size"],
    )
    samples = trainer.sample(n=n, shape=shape)         # (-1, 1)
    samples_01 = (samples * 0.5 + 0.5).clamp(0, 1)    # (0, 1)

    grid = vutils.make_grid(samples_01, nrow=8, padding=2)  # (C, H, W)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    if save_path is None:
        out_dir = Path(cfg["output"]["dir"]) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{cfg['run_name']}_grid.png"

    plt.figure(figsize=(12, 12 * grid_np.shape[0] / grid_np.shape[1]))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return (samples_01 * 255).to(torch.uint8)


def generate_comparison_grid(
    trainers: Dict[str, "SDDTrainer"],
    cfgs: Dict[str, Dict[str, Any]],
    n_per_variant: int = 8,
    seed: int = 0,
    save_path: str | Path | None = None,
) -> Path:
    """Generate side-by-side sample grids for multiple variants.

    Each row in the output image corresponds to one variant.

    Args:
        trainers: {variant_name: trainer}
        cfgs: {variant_name: config}
        n_per_variant: samples per variant (one row)
        seed: shared seed for fair comparison
        save_path: output PNG path

    Returns:
        Path to saved image
    """
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    rows = []
    for name, trainer in trainers.items():
        cfg = cfgs[name]
        torch.manual_seed(seed)
        shape = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
        samples = trainer.sample(n=n_per_variant, shape=shape)
        samples_01 = (samples * 0.5 + 0.5).clamp(0, 1)
        rows.append(samples_01)

    all_samples = torch.cat(rows, dim=0)  # (n_variants * n_per_variant, C, H, W)
    grid = vutils.make_grid(all_samples, nrow=n_per_variant, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    if save_path is None:
        out_dir = Path(list(cfgs.values())[0]["output"]["dir"]) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "comparison_grid.png"

    fig, ax = plt.subplots(figsize=(n_per_variant * 1.5, len(trainers) * 1.5))
    ax.imshow(grid_np)
    ax.set_yticks(
        [(i + 0.5) * grid_np.shape[0] / len(trainers) for i in range(len(trainers))]
    )
    ax.set_yticklabels(list(trainers.keys()), fontsize=11)
    ax.set_xticks([])
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return Path(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6: UMAP / t-SNE feature visualization (baseline vs SDD)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_for_viz(
    trainer,
    loader: DataLoader,
    feature_layer: str = "bottleneck",
    max_samples: int = 2000,
) -> tuple:
    """Extract features + labels, capped at max_samples for fast projection."""
    from src.evaluation.feature_extract import extract_features

    feats, labels = extract_features(trainer.model, loader, trainer.device, feature_layer=feature_layer)
    if feats.shape[0] > max_samples:
        idx = torch.randperm(feats.shape[0])[:max_samples]
        feats, labels = feats[idx], labels[idx]
    return feats.numpy(), labels.numpy()


def run_umap_comparison(
    trainers: Dict[str, "SDDTrainer"],
    loader: DataLoader,
    feature_layer: str = "bottleneck",
    max_samples: int = 2000,
    save_path: str | Path | None = None,
    class_names: list[str] | None = None,
) -> Path:
    """UMAP projection of features for each variant, plotted side by side.

    Args:
        trainers: {name: SDDTrainer}
        loader: DataLoader (typically test_loader)
        feature_layer: which layer to extract from
        max_samples: cap for UMAP speed
        save_path: output PNG path
        class_names: optional list of class name strings for legend

    Returns:
        Path to saved figure
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("pip install umap-learn") from e
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n = len(trainers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, trainer) in zip(axes, trainers.items()):
        feats_np, labels_np = extract_features_for_viz(trainer, loader, feature_layer, max_samples)
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(feats_np)
        n_classes = int(labels_np.max()) + 1
        colors = cm.get_cmap("tab10", n_classes)
        for cls in range(n_classes):
            mask = labels_np == cls
            label = class_names[cls] if class_names else str(cls)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.6,
                       color=colors(cls), label=label)
        ax.set_title(name, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        if class_names:
            ax.legend(markerscale=2, fontsize=7, loc="best")

    fig.suptitle(f"UMAP — feature layer: {feature_layer}", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = Path("outputs/figures/umap_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return Path(save_path)


def run_tsne_comparison(
    trainers: Dict[str, "SDDTrainer"],
    loader: DataLoader,
    feature_layer: str = "bottleneck",
    max_samples: int = 2000,
    save_path: str | Path | None = None,
    class_names: list[str] | None = None,
) -> Path:
    """t-SNE projection of features for each variant, plotted side by side.

    Same signature as run_umap_comparison; use when umap-learn is unavailable.
    """
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n = len(trainers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, trainer) in zip(axes, trainers.items()):
        feats_np, labels_np = extract_features_for_viz(trainer, loader, feature_layer, max_samples)
        embedding = TSNE(n_components=2, init="pca", learning_rate="auto",
                         perplexity=30, random_state=42).fit_transform(feats_np)
        n_classes = int(labels_np.max()) + 1
        colors = cm.get_cmap("tab10", n_classes)
        for cls in range(n_classes):
            mask = labels_np == cls
            label = class_names[cls] if class_names else str(cls)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.6,
                       color=colors(cls), label=label)
        ax.set_title(name, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        if class_names:
            ax.legend(markerscale=2, fontsize=7, loc="best")

    fig.suptitle(f"t-SNE — feature layer: {feature_layer}", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = Path("outputs/figures/tsne_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return Path(save_path)
