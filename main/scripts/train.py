from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.unet import UNetModel
from src.training.trainer import SDDTrainer
from src.utils.seed import set_seed


def build_loader(cfg):
    tfm = transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if cfg["dataset"]["name"] == "cifar10":
        ds = datasets.CIFAR10(root=cfg["dataset"]["root"], train=True, download=True, transform=tfm)
    else:
        ds = datasets.ImageFolder(root=cfg["dataset"]["train_dir"], transform=tfm)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"], pin_memory=True, drop_last=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["train"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetModel(
        base_channels=cfg["model"]["channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
        dropout=cfg["model"]["dropout"],
        image_size=cfg["dataset"]["image_size"],
    )
    trainer = SDDTrainer(model, cfg, device)
    loader = build_loader(cfg)
    params = list(trainer.model.parameters())
    if trainer.proj_student is not None:
        params += list(trainer.proj_student.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["mixed_precision"] and torch.cuda.is_available())
    for epoch in range(cfg["train"]["epochs"]):
        trainer.state.epoch = epoch
        trainer.train_one_epoch(loader, opt, scaler=scaler)

if __name__ == "__main__":
    main()
