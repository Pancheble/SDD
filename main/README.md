# Self-Distilled Diffusion Experiment Suite

This is a notebook-first research template for testing **Self-Distilled Diffusion (SDD)** on **CIFAR-10** and **Tiny ImageNet (64x64)** with a lightweight **UNet-based diffusion model**.

It is tuned for a single-GPU workstation such as an **RTX 4070**, with:
- mixed precision training
- gradient accumulation
- EMA teacher updates
- `tqdm` progress bars
- Weights & Biases logging
- ablation toggles for **centering**, **sharpening**, and **gating**

## Environment

The project pins PyTorch to the current stable 2.7.1 release line and matching torchvision/torchaudio wheels. Official PyTorch install pages list stable PyTorch 2.7.0/2.7.x wheels and the 2.7.1 GA announcement confirms 2.7.1 availability; torchvision 0.22.1 wheels are published on the official PyTorch wheel index. citeturn751335search0turn751335search5turn550293search3

Recommended install:
```bash
pip install -r requirements.txt
```

## Datasets

- **CIFAR-10**: fast ablation runs and sanity checks
- **Tiny ImageNet (64x64)**: stronger representation and generation experiments

The notebooks include dataset download and preparation cells for both datasets.

## Notebook workflow

The project is notebook-first. All experiment and visualization workflows live in `.ipynb` files:

- `01_prepare_data.ipynb`
- `02_train_baseline_cifar10.ipynb`
- `03_train_sdd_cifar10.ipynb`
- `04_train_sdd_tiny_imagenet.ipynb`
- `05_ablation_sweeps.ipynb`
- `06_eval_fid.ipynb`
- `07_eval_linear_probe.ipynb`
- `08_visualize_features.ipynb`
- `09_compare_runs.ipynb`
- `10_timestep_analysis.ipynb`

## Key ablations

The configs support:
- full SDD
- `w/o Centering`
- `w/o Sharpening`
- `w/o Gating`
- `w/o Projection Head`
- `MSE only`
- `EMA only`
- soft vs hard timestep gating
- temperature sweeps
- gating range sweeps
- EMA momentum sweeps
- distillation weight sweeps
- feature layer sweeps

## Project layout

```text
sdd-diffusion/
├── configs/
├── notebooks/
├── src/
├── scripts/
├── outputs/
└── data/
```

## How to run

Open the notebooks in order, starting from data preparation. Each training notebook exposes a config cell at the top so you can switch datasets or ablations without editing the source package.

## Notes

- The UNet is intentionally lightweight and RTX 4070-friendly.
- The trainer uses `tqdm` for live progress visualization.
- The wandb logger is optional and controlled through config, but enabled by default in the notebooks.
- Tiny ImageNet validation preparation is included as a notebook step because the raw dataset ships in a non-ImageFolder format.
