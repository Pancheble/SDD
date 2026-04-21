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


# ── config helpers ────────────────────────────────────────────────────────────

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


# ── dataset / dataloader ─────────────────────────────────────────────────────

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
    eval_tfm  = build_transforms(image_size, train=False)

    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=cfg["dataset"]["root"], train=True,  download=True, transform=train_tfm)
        test_ds  = datasets.CIFAR10(root=cfg["dataset"]["root"], train=False, download=True, transform=eval_tfm)
    elif name == "tiny_imagenet":
        train_dir = Path(cfg["dataset"].get("train_dir", "./data/tiny-imagenet-200/train"))
        val_dir   = Path(cfg["dataset"].get("val_dir",   "./data/tiny-imagenet-200/val"))
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                "Tiny ImageNet not found. Set train_dir / val_dir in the config."
            )
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfm)
        test_ds  = datasets.ImageFolder(root=str(val_dir),   transform=eval_tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_ds, test_ds


def build_loaders(cfg: Dict[str, Any], accelerator=None):
    """Build DataLoaders.

    When *accelerator* is provided the loader uses a DistributedSampler
    automatically (Accelerate handles this inside accelerator.prepare()).
    We skip manual shuffle/sampler here; Accelerate will re-wrap the loader.
    """
    train_ds, test_ds = build_datasets(cfg)
    batch_size   = cfg["train"]["batch_size"]
    num_workers  = cfg["dataset"].get("num_workers", 4)
    pin_memory   = cfg["dataset"].get("pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


# ── model / trainer builders ─────────────────────────────────────────────────

def build_model(cfg: Dict[str, Any]) -> UNetModel:
    return UNetModel(
        base_channels=cfg["model"]["channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
        dropout=cfg["model"]["dropout"],
        image_size=cfg["dataset"]["image_size"],
    )


def build_trainer(
    cfg: Dict[str, Any],
    device: str | torch.device | None = None,
    accelerator=None,
) -> SDDTrainer:
    model = build_model(cfg)
    return SDDTrainer(model, cfg, device=device, accelerator=accelerator)


def make_optimizer(trainer: SDDTrainer, cfg: Dict[str, Any]):
    params = list(trainer.model.parameters())
    if trainer.proj_student is not None:
        params += list(trainer.proj_student.parameters())
    return torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )


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


# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_checkpoint_path(cfg: Dict[str, Any], suffix: str = "last") -> Path:
    out_dir = Path(cfg["output"]["dir"]) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{cfg['run_name']}_{suffix}.pt"


def save_checkpoint(
    trainer: SDDTrainer,
    optimizer,
    cfg: Dict[str, Any],
    epoch: int,
    path: str | Path | None = None,
    accelerator=None,
) -> Path:
    """Save checkpoint — only executed on the main process."""
    path = Path(path) if path is not None else make_checkpoint_path(cfg, "last")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always save the unwrapped (non-DDP) model weights
    raw_model      = trainer._unwrap(trainer.model)
    raw_proj_s     = trainer._unwrap(trainer.proj_student) if trainer.proj_student is not None else None

    payload = {
        "model":       raw_model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "cfg":         cfg,
        "epoch":       epoch,
        "state":       trainer.state.__dict__,
        "teacher":     trainer.teacher.state_dict()     if trainer.teacher is not None else None,
        "proj_student": raw_proj_s.state_dict()         if raw_proj_s is not None else None,
        "proj_teacher": trainer.proj_teacher.state_dict() if trainer.proj_teacher is not None else None,
        "center":      trainer.center.value             if trainer.center is not None else None,
    }

    if accelerator is not None:
        accelerator.save(payload, path)
    else:
        torch.save(payload, path)
    return path


def load_checkpoint(
    trainer: SDDTrainer,
    optimizer=None,
    path: str | Path | None = None,
) -> dict:
    if path is None:
        raise ValueError("path is required to load a checkpoint")
    ckpt = torch.load(path, map_location=trainer.device, weights_only=False)

    trainer._unwrap(trainer.model).load_state_dict(ckpt["model"])
    if ckpt.get("teacher") is not None and trainer.teacher is not None:
        trainer.teacher.load_state_dict(ckpt["teacher"])
    if ckpt.get("proj_student") is not None and trainer.proj_student is not None:
        trainer._unwrap(trainer.proj_student).load_state_dict(ckpt["proj_student"])
    if ckpt.get("proj_teacher") is not None and trainer.proj_teacher is not None:
        trainer.proj_teacher.load_state_dict(ckpt["proj_teacher"])
    if ckpt.get("center") is not None and trainer.center is not None:
        trainer.center.value.copy_(ckpt["center"].to(trainer.device))
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "state" in ckpt:
        trainer.state.step  = ckpt["state"].get("step", 0)
        trainer.state.epoch = ckpt["state"].get("epoch", 0)
    return ckpt


def build_trainer_from_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: str | Path,
    device: str | None = None,
    accelerator=None,
) -> SDDTrainer:
    trainer = build_trainer(cfg, device=device, accelerator=accelerator)
    load_checkpoint(trainer, path=checkpoint_path)
    return trainer


# ── evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_generation(
    trainer: SDDTrainer,
    loader: DataLoader,
    cfg: Dict[str, Any],
    accelerator=None,
) -> Dict[str, float]:
    """Compute FID. In multi-GPU mode only runs on the main process."""
    if accelerator is not None and not accelerator.is_main_process:
        return {}

    raw_model = trainer._unwrap(trainer.model)
    raw_model.eval()
    device = trainer.device
    fid = FIDEvaluator(device=device)

    target = cfg["train"].get("fid_num_samples", 512)
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        x = (x * 0.5 + 0.5).clamp(0, 1)
        fid.update_real((x * 255).to(torch.uint8))
        seen += x.size(0)
        if seen >= target:
            break

    sample_shape = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
    remaining = target
    while remaining > 0:
        n    = min(remaining, cfg["train"].get("num_samples_preview", 64))
        fake = trainer.sample(n=n, shape=sample_shape)
        fake = (fake * 0.5 + 0.5).clamp(0, 1)
        fid.update_fake((fake * 255).to(torch.uint8))
        remaining -= n
    return {"fid": fid.compute()}


@torch.no_grad()
def collect_features(trainer: SDDTrainer, loader: DataLoader):
    return extract_features(trainer._unwrap(trainer.model), loader, trainer.device)


def run_linear_probe(
    trainer: SDDTrainer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
    accelerator=None,
) -> float:
    """Train a linear probe. In multi-GPU mode only runs on the main process."""
    if accelerator is not None and not accelerator.is_main_process:
        return float("nan")

    train_feats, train_labels = collect_features(trainer, train_loader)
    test_feats,  test_labels  = collect_features(trainer, test_loader)
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


# ── main training loop ────────────────────────────────────────────────────────

def train_epochs(
    trainer: SDDTrainer,
    train_loader: DataLoader,
    cfg: Dict[str, Any],
    optimizer,
    run=None,
    val_loader: DataLoader | None = None,
    accelerator=None,
) -> pd.DataFrame:
    """Epoch loop compatible with single-GPU (scaler) and Accelerate (no scaler)."""

    # Legacy single-GPU AMP scaler — only used when accelerator is None
    if accelerator is None:
        use_amp = cfg["train"]["mixed_precision"] and torch.cuda.is_available()
        scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None

    rows       = []
    save_every = cfg["train"].get("save_every", 10)
    last_ckpt  = None
    is_main    = (accelerator is None) or accelerator.is_main_process

    for epoch in range(cfg["train"]["epochs"]):
        trainer.state.epoch = epoch
        metrics = trainer.train_one_epoch(train_loader, optimizer, scaler=scaler, wandb_run=run)
        row = {"epoch": epoch, **metrics}
        rows.append(row)

        if is_main and run is not None:
            run.log({"epoch": epoch, **metrics})

        if val_loader is not None and (epoch + 1) % cfg["train"].get("eval_every", 10) == 0:
            val_metrics = evaluate_generation(trainer, val_loader, cfg, accelerator=accelerator)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if is_main and run is not None:
                run.log({**{f"val_{k}": v for k, v in val_metrics.items()}, "epoch": epoch})

        if is_main and ((epoch + 1) % save_every == 0 or epoch == cfg["train"]["epochs"] - 1):
            last_ckpt = save_checkpoint(trainer, optimizer, cfg, epoch, accelerator=accelerator)

    if not is_main:
        return pd.DataFrame(rows)

    history = pd.DataFrame(rows)
    out_dir = Path(cfg["output"]["dir"]) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / f"{cfg['run_name']}_history.csv", index=False)
    if last_ckpt is not None:
        history.attrs["checkpoint_path"] = str(last_ckpt)
    return history


# ── high-level experiment runners ─────────────────────────────────────────────

def run_experiment(
    cfg: Dict[str, Any],
    device: str | None = None,
    with_eval: bool = False,
    accelerator=None,
) -> dict:
    """Full train + optional eval. Supports both single-GPU and Accelerate."""
    is_main = (accelerator is None) or accelerator.is_main_process

    set_seed(cfg["train"]["seed"])
    train_loader, test_loader = build_loaders(cfg, accelerator=accelerator)
    trainer  = build_trainer(cfg, device=device, accelerator=accelerator)
    optimizer = make_optimizer(trainer, cfg)

    # Prepare optimizer (and loaders if not done in trainer._prepare)
    if accelerator is not None:
        optimizer, train_loader, test_loader = accelerator.prepare(
            optimizer, train_loader, test_loader
        )

    run = maybe_init_wandb(cfg) if is_main else None
    history = train_epochs(
        trainer, train_loader, cfg, optimizer,
        run=run,
        val_loader=test_loader if with_eval else None,
        accelerator=accelerator,
    )

    results: dict = {
        "history": history,
        "trainer": trainer,
        "train_loader": train_loader,
        "test_loader":  test_loader,
    }

    if with_eval and is_main:
        results["fid"] = evaluate_generation(trainer, test_loader, cfg, accelerator=accelerator)["fid"]
        results["linear_probe_acc"] = run_linear_probe(
            trainer, train_loader, test_loader, cfg, accelerator=accelerator
        )

    if run is not None:
        run.finish()
    return results


def load_and_evaluate(
    cfg: Dict[str, Any],
    checkpoint_path: str | Path,
    device: str | None = None,
    accelerator=None,
) -> dict:
    trainer = build_trainer_from_checkpoint(cfg, checkpoint_path, device=device, accelerator=accelerator)
    train_loader, test_loader = build_loaders(cfg)
    return {
        "fid": evaluate_generation(trainer, test_loader, cfg, accelerator=accelerator).get("fid"),
        "linear_probe_acc": run_linear_probe(trainer, train_loader, test_loader, cfg, accelerator=accelerator),
    }


def run_ablation_suite(
    base_cfg: Dict[str, Any],
    variants: Dict[str, Dict[str, Any]],
    device: str | None = None,
    accelerator=None,
) -> pd.DataFrame:
    outputs = []
    for name, overrides in variants.items():
        cfg = deep_update(base_cfg, overrides)
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_{name}"
        save_cfg(cfg, Path(base_cfg["output"]["dir"]) / "configs" / f"{cfg['run_name']}.yaml")
        result = run_experiment(cfg, device=device, with_eval=True, accelerator=accelerator)
        outputs.append({
            "variant": name,
            "fid": result.get("fid"),
            "linear_probe_acc": result.get("linear_probe_acc"),
        })
    return pd.DataFrame(outputs)


def run_timestep_sweep(
    base_cfg: Dict[str, Any],
    sweep: Iterable[tuple[float, float]],
    device: str | None = None,
    accelerator=None,
) -> pd.DataFrame:
    rows = []
    for t_min, t_max in sweep:
        cfg = deep_update(base_cfg, {"sdd": {"gating": {"t_min": t_min, "t_max": t_max}}})
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_gate_{t_min:.2f}_{t_max:.2f}"
        result = run_experiment(cfg, device=device, with_eval=True, accelerator=accelerator)
        rows.append({
            "t_min": t_min, "t_max": t_max,
            "fid": result.get("fid"),
            "linear_probe_acc": result.get("linear_probe_acc"),
        })
    return pd.DataFrame(rows)


# ── portfolio experiments (unchanged logic, accelerator-aware) ────────────────

FEATURE_LAYERS = ["bottleneck", "skip1", "skip2", "decoder1"]


def run_feature_layer_ablation(
    base_cfg: Dict[str, Any],
    layers: list[str] | None = None,
    device: str | None = None,
    accelerator=None,
) -> pd.DataFrame:
    layers = layers or FEATURE_LAYERS
    rows = []
    for layer in layers:
        cfg = deep_update(base_cfg, {"sdd": {"feature_layer": layer}})
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_feat_{layer}"
        save_cfg(cfg, Path(base_cfg["output"]["dir"]) / "configs" / f"{cfg['run_name']}.yaml")
        result = run_experiment(cfg, device=device, with_eval=True, accelerator=accelerator)
        rows.append({
            "feature_layer": layer,
            "fid": result.get("fid"),
            "linear_probe_acc": result.get("linear_probe_acc"),
        })
    return pd.DataFrame(rows)


def train_with_curves(
    cfg: Dict[str, Any],
    device: str | None = None,
    eval_every: int | None = None,
    accelerator=None,
) -> pd.DataFrame:
    is_main = (accelerator is None) or accelerator.is_main_process
    set_seed(cfg["train"]["seed"])
    train_loader, test_loader = build_loaders(cfg)
    trainer   = build_trainer(cfg, device=device, accelerator=accelerator)
    optimizer = make_optimizer(trainer, cfg)
    if accelerator is not None:
        optimizer, train_loader, test_loader = accelerator.prepare(
            optimizer, train_loader, test_loader
        )
    run = maybe_init_wandb(cfg) if is_main else None
    eval_every = eval_every or cfg["train"].get("eval_every", 10)

    use_amp = accelerator is None and cfg["train"]["mixed_precision"] and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp) if accelerator is None else None

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        trainer.state.epoch = epoch
        metrics = trainer.train_one_epoch(train_loader, optimizer, scaler=scaler, wandb_run=run)
        row: Dict[str, Any] = {"epoch": epoch, **metrics}

        if (epoch + 1) % eval_every == 0 or epoch == cfg["train"]["epochs"] - 1:
            gen_metrics = evaluate_generation(trainer, test_loader, cfg, accelerator=accelerator)
            probe_acc   = run_linear_probe(trainer, train_loader, test_loader, cfg, accelerator=accelerator)
            row["fid"]             = gen_metrics.get("fid")
            row["linear_probe_acc"] = probe_acc
            if is_main and run is not None:
                run.log({"epoch": epoch, "fid": gen_metrics.get("fid"), "linear_probe_acc": probe_acc})

        rows.append(row)
        if is_main and (epoch + 1) % cfg["train"].get("save_every", 10) == 0:
            save_checkpoint(trainer, optimizer, cfg, epoch, accelerator=accelerator)

    if run is not None:
        run.finish()

    history = pd.DataFrame(rows)
    if is_main:
        out_dir = Path(cfg["output"]["dir"]) / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_dir / f"{cfg['run_name']}_curve.csv", index=False)
    return history


def compare_training_curves(
    base_cfg: Dict[str, Any],
    variants: Dict[str, Dict[str, Any]],
    device: str | None = None,
    accelerator=None,
) -> Dict[str, pd.DataFrame]:
    results = {}
    for name, overrides in variants.items():
        cfg = deep_update(base_cfg, overrides)
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_{name}_curve"
        results[name] = train_with_curves(cfg, device=device, accelerator=accelerator)
    return results


def collect_gate_histogram(trainer, loader: DataLoader, n_batches: int = 50) -> Dict[str, Any]:
    import numpy as np
    from src.sdd.gating import timestep_gate

    cfg_gate = trainer.cfg["sdd"]["gating"]
    t_min    = cfg_gate["t_min"]
    t_max    = cfg_gate["t_max"]
    mode     = cfg_gate["mode"]
    soft_beta = cfg_gate.get("soft_beta", 0.08)
    T = trainer.diffusion.timesteps

    all_t, all_g = [], []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        t = torch.randint(0, T, (x.size(0),), device=trainer.device)
        g = timestep_gate(t, T, mode=mode, t_min=t_min, t_max=t_max, soft_beta=soft_beta)
        all_t.append(t.cpu().numpy())
        all_g.append(g.cpu().numpy())

    return {
        "timesteps":   np.concatenate(all_t),
        "gate_values": np.concatenate(all_g),
        "t_min": t_min, "t_max": t_max, "mode": mode, "T": T,
    }


def run_gating_analysis(
    base_cfg: Dict[str, Any],
    gating_configs: Dict[str, Dict[str, Any]] | None = None,
    device: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    if gating_configs is None:
        gating_configs = {
            "hard":        {"sdd": {"gating": {"mode": "hard"}}},
            "soft":        {"sdd": {"gating": {"mode": "soft", "soft_beta": 0.08}}},
            "soft_narrow": {"sdd": {"gating": {"mode": "soft", "soft_beta": 0.04}}},
        }
    train_loader, _ = build_loaders(base_cfg)
    results = {}
    for name, overrides in gating_configs.items():
        cfg     = deep_update(base_cfg, overrides)
        trainer = build_trainer(cfg, device=device)
        results[name] = collect_gate_histogram(trainer, train_loader)
    return results


def run_ema_momentum_sweep(
    base_cfg: Dict[str, Any],
    momentum_values: list[float] | None = None,
    device: str | None = None,
    accelerator=None,
) -> pd.DataFrame:
    momentum_values = momentum_values or [0.990, 0.996, 0.999]
    rows = []
    for m in momentum_values:
        cfg = deep_update(base_cfg, {"sdd": {"teacher_momentum": m}})
        cfg["run_name"] = f"{base_cfg['dataset']['name']}_ema_{str(m).replace('.', '')}"
        save_cfg(cfg, Path(base_cfg["output"]["dir"]) / "configs" / f"{cfg['run_name']}.yaml")
        result = run_experiment(cfg, device=device, with_eval=True, accelerator=accelerator)
        rows.append({
            "teacher_momentum": m,
            "fid": result.get("fid"),
            "linear_probe_acc": result.get("linear_probe_acc"),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def generate_sample_grid(trainer, cfg, n=64, seed=0, save_path=None):
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    torch.manual_seed(seed)
    shape   = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
    samples = trainer.sample(n=n, shape=shape)
    samples_01 = (samples * 0.5 + 0.5).clamp(0, 1)
    grid    = vutils.make_grid(samples_01, nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if save_path is None:
        out_dir = Path(cfg["output"]["dir"]) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{cfg['run_name']}_grid.png"
    plt.figure(figsize=(12, 12 * grid_np.shape[0] / grid_np.shape[1]))
    plt.imshow(grid_np); plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return (samples_01 * 255).to(torch.uint8)


def generate_comparison_grid(trainers, cfgs, n_per_variant=8, seed=0, save_path=None):
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    rows = []
    for name, trainer in trainers.items():
        cfg = cfgs[name]
        torch.manual_seed(seed)
        shape = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
        samples = trainer.sample(n=n_per_variant, shape=shape)
        rows.append((samples * 0.5 + 0.5).clamp(0, 1))
    all_samples = torch.cat(rows, dim=0)
    grid    = vutils.make_grid(all_samples, nrow=n_per_variant, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if save_path is None:
        out_dir = Path(list(cfgs.values())[0]["output"]["dir"]) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "comparison_grid.png"
    fig, ax = plt.subplots(figsize=(n_per_variant * 1.5, len(trainers) * 1.5))
    ax.imshow(grid_np)
    ax.set_yticks([(i + 0.5) * grid_np.shape[0] / len(trainers) for i in range(len(trainers))])
    ax.set_yticklabels(list(trainers.keys()), fontsize=11); ax.set_xticks([])
    plt.tight_layout(pad=0.5); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return Path(save_path)


def extract_features_for_viz(trainer, loader, feature_layer="bottleneck", max_samples=2000):
    feats, labels = extract_features(
        trainer._unwrap(trainer.model), loader, trainer.device, feature_layer=feature_layer
    )
    if feats.shape[0] > max_samples:
        idx = torch.randperm(feats.shape[0])[:max_samples]
        feats, labels = feats[idx], labels[idx]
    return feats.numpy(), labels.numpy()


def run_umap_comparison(trainers, loader, feature_layer="bottleneck",
                        max_samples=2000, save_path=None, class_names=None):
    try:
        import umap
    except ImportError as e:
        raise ImportError("pip install umap-learn") from e
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n = len(trainers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, trainer) in zip(axes, trainers.items()):
        f, l = extract_features_for_viz(trainer, loader, feature_layer, max_samples)
        emb  = umap.UMAP(n_components=2, random_state=42).fit_transform(f)
        colors = cm.get_cmap("tab10", int(l.max()) + 1)
        for cls in range(int(l.max()) + 1):
            mask = l == cls
            ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.6, color=colors(cls),
                       label=class_names[cls] if class_names else str(cls))
        ax.set_title(name, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
        if class_names:
            ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.suptitle(f"UMAP — {feature_layer}", fontsize=14); plt.tight_layout()
    if save_path is None:
        save_path = Path("outputs/figures/umap_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return Path(save_path)


def run_tsne_comparison(trainers, loader, feature_layer="bottleneck",
                        max_samples=2000, save_path=None, class_names=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n = len(trainers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, trainer) in zip(axes, trainers.items()):
        f, l = extract_features_for_viz(trainer, loader, feature_layer, max_samples)
        emb  = TSNE(n_components=2, init="pca", learning_rate="auto",
                    perplexity=30, random_state=42).fit_transform(f)
        colors = cm.get_cmap("tab10", int(l.max()) + 1)
        for cls in range(int(l.max()) + 1):
            mask = l == cls
            ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.6, color=colors(cls),
                       label=class_names[cls] if class_names else str(cls))
        ax.set_title(name, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
        if class_names:
            ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.suptitle(f"t-SNE — {feature_layer}", fontsize=14); plt.tight_layout()
    if save_path is None:
        save_path = Path("outputs/figures/tsne_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return Path(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# notebook_launcher helpers
# Wrap long-running jobs so they can be launched from a .ipynb cell with
# multi-GPU support via accelerate.notebook_launcher.
#
# num_processes=None → auto-detect torch.cuda.device_count() (≥1)
# num_processes=1    → single-process, no DDP overhead
# ─────────────────────────────────────────────────────────────────────────────

def _get_num_processes(num_processes: int | None) -> int:
    import torch
    if num_processes is not None:
        return num_processes
    n = torch.cuda.device_count()
    return max(n, 1)


def _train_fn(cfg, with_eval):
    """Worker function executed inside notebook_launcher."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed as acc_set_seed
    accelerator = Accelerator(mixed_precision="fp16" if _get_num_processes(None) > 0 else "no")
    acc_set_seed(cfg["train"]["seed"])
    train_loader, test_loader = build_loaders(cfg)
    trainer   = build_trainer(cfg, accelerator=accelerator)
    optimizer = make_optimizer(trainer, cfg)
    optimizer, train_loader, test_loader = accelerator.prepare(
        optimizer, train_loader, test_loader
    )
    run = maybe_init_wandb(cfg) if accelerator.is_main_process else None
    train_epochs(
        trainer, train_loader, cfg, optimizer,
        run=run,
        val_loader=test_loader if with_eval else None,
        accelerator=accelerator,
    )
    accelerator.wait_for_everyone()
    if with_eval and accelerator.is_main_process:
        evaluate_generation(trainer, test_loader, cfg, accelerator=accelerator)
        run_linear_probe(trainer, train_loader, test_loader, cfg, accelerator=accelerator)
    if run is not None:
        run.finish()
    accelerator.end_training()


def launch_train(
    cfg: Dict[str, Any],
    num_processes: int | None = None,
    with_eval: bool = False,
) -> None:
    """Launch training from a Jupyter notebook cell.

    Args:
        cfg: experiment config dict (use load_cfg + deep_update to build it)
        num_processes: number of GPUs to use; None = auto-detect
        with_eval: run FID + linear probe after training

    Example::

        cfg = load_cfg("configs/cifar10.yaml")
        launch_train(cfg, with_eval=True)
    """
    from accelerate import notebook_launcher
    n = _get_num_processes(num_processes)
    print(f"[launch_train] starting on {n} process(es) ...")
    notebook_launcher(_train_fn, args=(cfg, with_eval), num_processes=n)
    print("[launch_train] done.")


def _eval_fid_fn(cfg, checkpoint_path, num_samples):
    from accelerate import Accelerator
    import torch
    accelerator = Accelerator()
    trainer = build_trainer_from_checkpoint(cfg, checkpoint_path, accelerator=accelerator)
    trainer._unwrap(trainer.model).eval()
    _, test_loader = build_loaders(cfg)
    test_loader = accelerator.prepare(test_loader)

    from src.evaluation.fid import FIDEvaluator
    fid = FIDEvaluator(device=accelerator.device) if accelerator.is_main_process else None

    seen = 0
    for x, _ in test_loader:
        x_all = accelerator.gather(x)
        if accelerator.is_main_process:
            x_01 = (x_all * 0.5 + 0.5).clamp(0, 1)
            fid.update_real((x_01 * 255).to(torch.uint8))
            seen += x_01.size(0)
        if seen >= num_samples:
            break

    accelerator.wait_for_everyone()
    samples_per_proc = (num_samples + accelerator.num_processes - 1) // accelerator.num_processes
    sample_shape = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
    generated, remaining = [], samples_per_proc
    while remaining > 0:
        n = min(remaining, cfg["train"].get("num_samples_preview", 64))
        fake = trainer.sample(n=n, shape=sample_shape)
        generated.append((fake * 0.5 + 0.5).clamp(0, 1))
        remaining -= n
    fake_all = accelerator.gather(torch.cat(generated, dim=0))
    if accelerator.is_main_process:
        fid.update_fake((fake_all * 255).to(torch.uint8))
        score = fid.compute()
        print(f"[eval_fid] FID = {score:.4f}  ({num_samples} samples, {accelerator.num_processes} GPU(s))")


def launch_eval_fid(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    num_samples: int = 2048,
    num_processes: int | None = None,
) -> None:
    """Launch FID evaluation from a Jupyter notebook cell.

    Example::

        launch_eval_fid(cfg, "outputs/checkpoints/cifar10_full_sdd_last.pt")
    """
    from accelerate import notebook_launcher
    n = _get_num_processes(num_processes)
    print(f"[launch_eval_fid] starting on {n} process(es) ...")
    notebook_launcher(_eval_fid_fn, args=(cfg, checkpoint_path, num_samples), num_processes=n)
    print("[launch_eval_fid] done.")


def _eval_linear_fn(cfg, checkpoint_path, probe_epochs, probe_lr):
    from accelerate import Accelerator
    import torch
    accelerator = Accelerator()
    trainer = build_trainer_from_checkpoint(cfg, checkpoint_path, accelerator=accelerator)
    raw_model = trainer._unwrap(trainer.model)
    raw_model.eval()

    train_loader, test_loader = build_loaders(cfg)
    train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

    def _extract(loader):
        all_f, all_l = [], []
        with torch.no_grad():
            for x, y in loader:
                t = torch.zeros(x.size(0), device=accelerator.device, dtype=torch.long)
                _, f = raw_model(x, t, return_features=True)
                all_f.append(accelerator.gather(f).cpu())
                all_l.append(accelerator.gather(y).cpu())
        return torch.cat(all_f), torch.cat(all_l)

    train_feats, train_labels = _extract(train_loader)
    test_feats,  test_labels  = _extract(test_loader)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        from src.evaluation.linear_probe import train_linear_probe
        _, acc = train_linear_probe(
            train_feats=train_feats, train_labels=train_labels,
            val_feats=test_feats,   val_labels=test_labels,
            num_classes=cfg["dataset"]["num_classes"],
            epochs=probe_epochs, lr=probe_lr,
            device=str(accelerator.device),
        )
        print(f"[eval_linear] linear probe accuracy = {acc:.4f}")


def launch_eval_linear(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    probe_epochs: int = 50,
    probe_lr: float = 1e-3,
    num_processes: int | None = None,
) -> None:
    """Launch linear probe evaluation from a Jupyter notebook cell.

    Example::

        launch_eval_linear(cfg, "outputs/checkpoints/cifar10_full_sdd_last.pt")
    """
    from accelerate import notebook_launcher
    n = _get_num_processes(num_processes)
    print(f"[launch_eval_linear] starting on {n} process(es) ...")
    notebook_launcher(_eval_linear_fn, args=(cfg, checkpoint_path, probe_epochs, probe_lr), num_processes=n)
    print("[launch_eval_linear] done.")
