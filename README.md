# Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models

**[English](README.md) | [Korean](README.ko.md)**

---

## Abstract

Diffusion models have achieved remarkable success in generative modeling, yet their internal representations remain poorly structured for downstream discriminative tasks. The standard denoising objective — a mean squared error regression in pixel space — produces smooth, low-discriminability features that underperform compared to dedicated self-supervised representation learners such as DINO. In this work, we propose **Self-Distilled Diffusion (SDD)**, a training framework that augments the standard diffusion objective with a DINO-style self-distillation loss applied directly to UNet intermediate features. Concretely, we maintain an EMA-updated teacher network alongside the student diffusion model, project intermediate features into a shared embedding space, and apply centering and sharpening operations to produce stable, confident training targets. Unlike prior work such as REPA, our method requires no external pretrained encoder and is fully self-contained. We further introduce a timestep-adaptive weighting scheme that restricts the distillation signal to the low-to-middle noise regime, where semantic features are most active.

---

## 1. Introduction

Denoising diffusion probabilistic models (DDPMs) have established themselves as the dominant paradigm for high-fidelity image synthesis. Their standard training objective is:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

where $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1 - \bar{\alpha}\_t} \epsilon$ is the noisy image at timestep $t$, and $\epsilon_\theta$ is a neural network (typically a UNet or DiT) learning to predict the added noise.

This regression objective has a fundamental limitation: it provides no explicit incentive for the model to build structured, semantically meaningful internal representations. Recent empirical studies have confirmed that diffusion models trained solely with this objective produce representations significantly inferior to those of dedicated discriminative models such as DINOv2, even after extended training.

Meanwhile, DINO (Self-Distillation with No Labels) demonstrated that a teacher-student self-distillation framework, augmented with two simple operations — **centering** (subtracting a running mean to prevent collapse) and **sharpening** (applying a lower temperature to the teacher to produce confident targets) — yields exceptionally strong visual representations without any labeled data.

We ask the natural question: **can DINO's centering and sharpening be applied directly to diffusion model features, enabling the model to simultaneously denoise and learn structured representations, without relying on any external pretrained encoder?**

Our contributions are:

1. We propose **Self-Distilled Diffusion (SDD)**, a framework that applies DINO-style centering and sharpening within a fully self-contained diffusion training loop, requiring no external encoders.
2. We introduce a **timestep-adaptive gating mechanism** that selectively applies the distillation loss only in the noise regime where semantic features are most active, resolving the feature-timestep conflict inherent to naive application.
3. We design a systematic ablation to verify that both centering and sharpening contribute independently to representation quality, and that their combination is superior to EMA-based distillation without these operations (as used in prior work such as SD-DiT and DDAE).

---

## 2. Related Work

### 2.1 Diffusion Models

DDPMs introduced a principled framework for learning data distributions via iterative denoising. Score-based generative models and the subsequent DDIM sampler established continuous-time and deterministic variants. Architecture advances — including ADM, LDM, and DiT — scaled diffusion models to state-of-the-art generation quality. However, all of these methods share the same pixel-space MSE objective, with no mechanism for representation shaping.

### 2.2 Self-Supervised Representation Learning

MoCo, SimCLR, and BYOL established contrastive and self-distillation frameworks for discriminative representation learning. DINO extended this paradigm with a teacher-student architecture featuring centering and sharpening, producing representations that exhibit emergent segmentation properties without any labels. DINOv2 scaled this to large pretrained models. This project aims to fill this gap by porting the centering+sharpening mechanism into a generative diffusion training loop.

### 2.3 Representation Learning via Diffusion

Several works have noted that diffusion models incidentally learn useful representations. DDAE explored using diffusion models as representation learners via EMA distillation. REPA (ICLR 2025 Oral) proposed aligning diffusion model features to an external DINOv2 encoder during training, achieving dramatic speedups and improved generation quality. SRA and SD-DiT explored internal EMA teacher-student structures within diffusion training but did not apply centering or sharpening. This project fills this gap by applying DINO's full centering + sharpening mechanism in a self-contained, external-encoder-free diffusion training loop, with an explicit ablation design showing that these operations matter.

---

## 3. Method

### 3.1 Preliminaries: Standard Diffusion Training

Given a data distribution $q(x_0)$, the forward process gradually adds Gaussian noise:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

The model $\epsilon_\theta(x_t, t)$ is trained to reverse this process by minimizing:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 3.2 DINO Background

DINO trains a student network $f_s$ and an EMA-updated teacher network $f_t$ (with momentum $m$) on augmented views of the same image. The key operations are:

**Centering** — subtracts a running mean $c$ from the teacher output before softmax, preventing collapse to trivial solutions:
$$p_t = \text{softmax}\left(\frac{z_t - c}{\tau_t}\right), \quad c \leftarrow \alpha \cdot c + (1 - \alpha) \cdot \mathbb{E}_{\text{batch}}[z_t]$$

**Sharpening** — applies a lower temperature $\tau_t \ll \tau_s$ to the teacher, producing confident (low-entropy) target distributions even when the student output is uncertain.

The loss is a cross-entropy between the teacher distribution and the student distribution:
$$\mathcal{L}_{\text{DINO}} = -\sum_k p_t^{(k)} \log p_s^{(k)}$$

### 3.3 Self-Distilled Diffusion (SDD)

#### 3.3.1 Architecture

We maintain two networks:

- **Student**: $\epsilon_\theta$ — the standard diffusion UNet/DiT, updated by gradient descent.
- **Teacher**: $\epsilon_\xi$ — an EMA copy of the student, updated as $\xi \leftarrow m \cdot \xi + (1-m) \cdot \theta$, with momentum $m = 0.996$.

Both networks receive independently augmented views of the same image $x_0$ (e.g., random crop, color jitter), each then corrupted through the forward diffusion process to produce their respective noisy inputs $x_t$.

#### 3.3.2 Feature Extraction and Projection

We extract intermediate features from the bottleneck layer of the UNet (or the middle transformer blocks of a DiT). Let $z_s = f_s(x_t, t)$ and $z_t = f_t(x_t, t)$ denote the student and teacher bottleneck features respectively.

Each is passed through a lightweight 2-layer MLP projection head $g(\cdot)$ mapping to a $K$-dimensional embedding (we use $K = 256$ by default):

$$\tilde{z}_s = g_s(z_s), \quad \tilde{z}_t = g_t(z_t)$$

The projection heads are not shared; the teacher head $g_t$ is also EMA-updated.

#### 3.3.3 Centering and Sharpening

Following DINO exactly, we apply centering to the teacher logits and a lower temperature, and a higher temperature to the student:

$$p_t = \text{softmax}\left(\frac{\tilde{z}_t - c}{\tau_t}\right), \quad p_s = \text{softmax}\left(\frac{\tilde{z}_s}{\tau_s}\right)$$

where:
- $c$ is updated each batch as a running EMA of teacher outputs: $c \leftarrow \alpha_c \cdot c + (1-\alpha_c) \mathbb{E}[\tilde{z}_t]$
- $\tau_t = 0.04$ (teacher temperature, low → sharpens the target distribution)
- $\tau_s = 0.1$ (student temperature, higher → softer student predictions)
- $\alpha_c = 0.9$ (centering EMA momentum)

#### 3.3.4 Timestep-Adaptive Gating

A naive application of the distillation loss across all timesteps is problematic: at high noise levels ($t \to T$), the input is nearly pure Gaussian noise and contains no semantic signal, making the distillation target meaningless. We introduce a **timestep gate**:

$$w(t) = \mathbb{1}\left[t_{\min} \leq t \leq t_{\max}\right]$$

where we set $t_{\min} = 0.1T$ and $t_{\max} = 0.6T$ by default. This restricts the distillation signal to the low-to-middle noise regime, where prior work has shown that semantic features are most active in diffusion model internals.

Optionally, a soft sigmoid gate can replace the hard threshold:

$$w(t) = \sigma\left(-\frac{t - t_{\text{mid}}}{\beta}\right)$$

where $t_{\text{mid}} = 0.4T$ and $\beta$ controls the sharpness of the gate.

#### 3.3.5 Total Training Objective

The full SDD objective is:

$$\mathcal{L}_{\text{SDD}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot w(t) \cdot \mathcal{L}_{\text{DINO}}(p_t, p_s)$$

where:

$$\mathcal{L}_{\text{MSE}} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

$$\mathcal{L}_{\text{DINO}} = -\sum_{k=1}^K p_t^{(k)} \log p_s^{(k)}$$

and $\gamma = 0.5$ is a scalar balancing the two objectives. Gradients flow only through the student; the teacher is updated by EMA only.

---

## References

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*.

[2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*.

[3] Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *ICLR 2021*. (DDIM)

[4] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *ICCV 2021*. (DINO)

[5] Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning robust visual features without supervision. *TMLR 2024*.

[6] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022*. (LDM)

[7] Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023*. (DiT)

[8] Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS 2021*. (ADM)

[9] Chen, X., et al. (2023). DDAE: Towards self-supervised representation learning with diffusion autoencoders. *arXiv 2023*.

[10] Yu, S., Kwak, D., Jang, H., et al. (2024). Representation alignment for generation: Training diffusion transformers is easier than you think. *ICLR 2025 Oral*. (REPA)

[11] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR 2020*. (MoCo)

[12] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*. (SimCLR)

[13] Grill, J. B., Strub, F., Altché, F., et al. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS 2020*. (BYOL)
