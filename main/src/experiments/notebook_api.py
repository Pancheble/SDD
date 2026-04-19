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
