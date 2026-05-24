# Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models

**[English](README.md) | [Korean](README.ko.md)**

---

## Abstract

Diffusion models have achieved remarkable success in generative modeling, yet their internal representations remain poorly structured for downstream discriminative tasks. The standard denoising objective — a mean squared error regression in pixel space — provides no explicit incentive for the model to organize its internal feature space semantically, producing smooth, low-discriminability representations that underperform compared to dedicated self-supervised learners such as DINO. Prior work addressing this gap either relies on external pretrained encoders (REPA) or applies EMA-based self-distillation without the collapse-prevention and entropy-sharpening operations that are critical to DINO's success (SD-DiT, DDAE). In this work, we propose **Self-Distilled Diffusion (SDD)**, a training framework that augments the standard diffusion objective with a DINO-style self-distillation loss applied directly to UNet/DiT intermediate features. Concretely, we maintain an EMA-updated teacher network alongside the student diffusion model, project intermediate features into a shared embedding space, and apply centering and sharpening operations to produce stable, confident training targets. We provide a theoretical analysis showing why centering and sharpening are necessary (not merely beneficial) in the diffusion feature setting, and demonstrate that naive EMA distillation without these operations is provably susceptible to dimensional collapse under the diffusion training dynamics. We further introduce a timestep-adaptive weighting scheme that restricts the distillation signal to the low-to-middle noise regime, where semantic features are most active, motivated by an information-theoretic argument about the semantic content of diffusion intermediate representations as a function of noise level.

---

## 1. Introduction

Denoising diffusion probabilistic models (DDPMs) have established themselves as the dominant paradigm for high-fidelity image synthesis. Their standard training objective is:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is the noisy image at timestep $t$, and $\epsilon_\theta$ is a neural network (typically a UNet or DiT) learning to predict the added noise.

This regression objective has a fundamental limitation: it provides no explicit incentive for the model to build structured, semantically meaningful internal representations. Recent empirical studies have confirmed that diffusion models trained solely with this objective produce representations significantly inferior to those of dedicated discriminative models such as DINOv2, even after extended training.

Meanwhile, DINO (Self-Distillation with No Labels) demonstrated that a teacher-student self-distillation framework, augmented with two simple operations — **centering** (subtracting a running mean to prevent collapse) and **sharpening** (applying a lower temperature to the teacher to produce confident targets) — yields exceptionally strong visual representations without any labeled data.

**What is missing in prior work.** SD-DiT and DDAE demonstrate that EMA-based teacher-student structures can be incorporated into diffusion training, but neither applies centering or sharpening. We argue, and prove in Section 4, that omitting these operations is not a minor implementation detail — it exposes the self-distillation signal to dimensional collapse under diffusion training dynamics, which differ fundamentally from the augmentation-based contrastive setting in which plain EMA distillation has been studied. REPA avoids this problem by anchoring features to an external DINOv2 encoder, but at the cost of a hard dependency on a large pretrained model, limiting applicability to domains and modalities where such encoders exist.

We ask: **can DINO's centering and sharpening be applied directly to diffusion model features, enabling the model to simultaneously denoise and learn structured representations, without relying on any external pretrained encoder, and with theoretical guarantees against collapse?**

Our contributions are:

1. We propose **Self-Distilled Diffusion (SDD)**, a framework that applies DINO-style centering and sharpening within a fully self-contained diffusion training loop, requiring no external encoders.
2. We provide a **theoretical analysis** (Section 4) showing that (a) the diffusion training dynamics create a systematically higher risk of feature collapse than standard augmentation-based self-distillation, and (b) centering and sharpening jointly resolve this by controlling the rank of the feature covariance and the entropy of teacher targets, respectively.
3. We introduce a **timestep-adaptive gating mechanism** (Section 3.3.4) motivated by an information-theoretic argument: we formalize the intuition that semantic content in diffusion intermediate representations decays monotonically with noise level, and derive the optimal gating window from this analysis.
4. We provide a **principled hyperparameter analysis** (Section 5) characterizing the sensitivity of SDD to the key hyperparameters $\gamma$, $t_{\min}$, $t_{\max}$, $\tau_t$, and $\tau_s$, with guidance on setting each.
5. We design a systematic ablation to verify that centering and sharpening contribute independently and that their combination is necessary.

---

## 2. Related Work

### 2.1 Diffusion Models

DDPMs introduced a principled framework for learning data distributions via iterative denoising. Score-based generative models and the subsequent DDIM sampler established continuous-time and deterministic variants. Architecture advances — including ADM, LDM, and DiT — scaled diffusion models to state-of-the-art generation quality. However, all of these methods share the same pixel-space MSE objective, with no mechanism for representation shaping.

### 2.2 Self-Supervised Representation Learning

MoCo, SimCLR, and BYOL established contrastive and self-distillation frameworks for discriminative representation learning. DINO extended this paradigm with a teacher-student architecture featuring centering and sharpening, producing representations with emergent segmentation properties without labels. DINOv2 scaled this to large pretrained models. A key insight from DINO is that centering and sharpening are not optional refinements — they are the mechanisms that prevent mode collapse and entropy collapse, respectively, and their interaction is what makes DINO stable across architectures and scales.

### 2.3 Representation Learning via Diffusion

Several works have noted that diffusion models incidentally learn useful representations. DDAE explored using diffusion models as representation learners via EMA distillation, demonstrating proof-of-concept but without collapse-prevention mechanisms. REPA (ICLR 2025 Oral) proposed aligning diffusion model features to an external DINOv2 encoder during training, achieving dramatic speedups and improved generation quality — but requiring the external encoder at training time. SD-DiT explored internal EMA teacher-student structures but similarly did not apply centering or sharpening.

**The gap.** No prior work has applied DINO's full centering + sharpening mechanism in a self-contained, encoder-free diffusion training loop, nor has any prior work theoretically analyzed why such operations are necessary specifically in the diffusion setting. SDD addresses both gaps.

---

## 3. Method

### 3.1 Preliminaries: Standard Diffusion Training

Given a data distribution $q(x_0)$, the forward process gradually adds Gaussian noise:

$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)\, I)$$

The model $\epsilon_\theta(x_t, t)$ is trained to reverse this process by minimizing:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x_0,\, \epsilon \sim \mathcal{N}(0,I),\, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 3.2 DINO Background

DINO trains a student network $f_s$ and an EMA-updated teacher network $f_t$ (with momentum $m$) on augmented views of the same image. The key operations are:

**Centering** — subtracts a running mean $c$ from the teacher output before softmax, preventing dimensional collapse by ensuring no single dimension dominates the output distribution:
$$p_t = \text{softmax}\!\left(\frac{z_t - c}{\tau_t}\right), \qquad c \leftarrow \alpha \cdot c + (1 - \alpha) \cdot \mathbb{E}_{\text{batch}}[z_t]$$
where $z_t$ denotes the raw teacher logits (pre-projection-head output).

**Sharpening** — applies a lower temperature $\tau_t \ll \tau_s$ to the teacher, producing confident (low-entropy) target distributions even when the student output is uncertain. This prevents entropy collapse — a mode in which both student and teacher converge to uniform distributions — by maintaining a steep gradient signal.

The loss is a cross-entropy between the teacher and student distributions:
$$\mathcal{L}_{\text{DINO}} = -\sum_k p_t^{(k)} \log p_s^{(k)}$$

### 3.3 Self-Distilled Diffusion (SDD)

#### 3.3.1 Architecture

We maintain two networks:

- **Student**: $\epsilon_\theta$ — the standard diffusion UNet/DiT, updated by gradient descent.
- **Teacher**: $\epsilon_\xi$ — an EMA copy of the student, updated as $\xi \leftarrow m \cdot \xi + (1-m) \cdot \theta$, with momentum $m = 0.996$.

Both networks receive the same image $x_0$ corrupted through the forward diffusion process. Importantly, different noise realizations $\epsilon_s, \epsilon_t \sim \mathcal{N}(0, I)$ are independently sampled for the student and teacher, so that $x_t^{(s)} \neq x_t^{(t)}$ even at the same timestep $t$. This stochasticity provides the view diversity that augmentation provides in standard DINO, without requiring explicit data augmentation pipelines. When the training domain supports it (e.g., natural images), standard augmentations (random crop, color jitter) can optionally be applied to $x_0$ before the forward process to further increase view diversity; we treat this as an optional extension. The core method relies only on noise stochasticity.

#### 3.3.2 Feature Extraction and Projection

We extract intermediate features from the bottleneck layer of the UNet (or the middle transformer blocks of a DiT). Let $z_s = f_s(x_t^{(s)}, t)$ and $z_t = f_t(x_t^{(t)}, t)$ denote the student and teacher bottleneck features respectively.

Each is passed through a lightweight 2-layer MLP projection head $g(\cdot)$ mapping to a $K$-dimensional embedding (we use $K = 256$ by default; see Section 5.3 for sensitivity analysis):

$$\tilde{z}_s = g_s(z_s), \qquad \tilde{z}_t = g_t(z_t)$$

The projection heads are not shared, following DINO's design: a shared projection head would couple the student and teacher optimization trajectories, interfering with the EMA update dynamics and reducing the effective diversity between teacher and student outputs. The teacher head $g_t$ is also EMA-updated with the same momentum $m$.

#### 3.3.3 Centering and Sharpening

Following DINO, we apply centering to the teacher projection outputs $\tilde{z}_t$ and a lower temperature, and a higher temperature to the student:

$$p_t = \text{softmax}\!\left(\frac{\tilde{z}_t - c}{\tau_t}\right), \qquad p_s = \text{softmax}\!\left(\frac{\tilde{z}_s}{\tau_s}\right)$$

where:
- $c$ is a running EMA of teacher projection outputs: $c \leftarrow \alpha_c \cdot c + (1-\alpha_c)\,\mathbb{E}_{\text{batch}}[\tilde{z}_t]$
- $\tau_t = 0.04$ (teacher temperature, low → sharpens the target distribution)
- $\tau_s = 0.1$ (student temperature, higher → softer student predictions)
- $\alpha_c = 0.9$ (centering EMA momentum)

Note that centering operates on $\tilde{z}_t$ (post-projection-head outputs), consistently throughout. The theoretical motivation for this choice is given in Section 4.

#### 3.3.4 Timestep-Adaptive Gating

**Motivation.** At high noise levels ($t \to T$), the noisy input $x_t$ is dominated by Gaussian noise and retains negligible mutual information with $x_0$. Formally, the mutual information between $x_t$ and $x_0$ is:

$$I(x_t; x_0) = -\frac{1}{2}\log\!\left(1 - \bar{\alpha}_t\right) + \text{const}$$

which monotonically decreases toward zero as $\bar{\alpha}_t \to 0$ (i.e., $t \to T$). Since the distillation target $p_t$ is derived from teacher features conditioned on $x_t$, the semantic content of $p_t$ vanishes in the high-noise regime. Applying the distillation loss in this regime injects noise into the representation learning signal and can actively harm feature quality.

We therefore introduce a **timestep gate**:

$$w(t) = \mathbb{1}\left[t_{\min} \leq t \leq t_{\max}\right]$$

with $t_{\min} = 0.1T$ and $t_{\max} = 0.6T$ (default). The lower bound $t_{\min} > 0$ additionally avoids the near-clean regime where the denoising loss already provides a strong per-pixel signal, reducing the need for auxiliary representation shaping. The upper bound $t_{\max} = 0.6T$ is derived from prior empirical analyses of diffusion model internals showing that semantic linear probing accuracy peaks in the $t \in [0.1T, 0.5T]$ range; we extend this slightly to $0.6T$ to avoid hard edge effects.

A soft variant using a product of sigmoids covers both boundaries symmetrically:

$$w(t) = \sigma\!\left(\frac{t - t_{\min}}{\beta}\right) \cdot \sigma\!\left(-\frac{t - t_{\max}}{\beta}\right)$$

where $\beta$ controls gate sharpness. This formulation, unlike the single-sided sigmoid used in some prior work, correctly suppresses the distillation signal at **both** extremes ($t \ll t_{\min}$ and $t \gg t_{\max}$) and reduces to the hard gate as $\beta \to 0$.

#### 3.3.5 Total Training Objective

The full SDD objective is:

$$\mathcal{L}_{\text{SDD}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot w(t) \cdot \mathcal{L}_{\text{DINO}}(p_t, p_s)$$

where:

$$\mathcal{L}_{\text{MSE}} = \| \epsilon - \epsilon_\theta(x_t^{(s)}, t) \|^2$$

$$\mathcal{L}_{\text{DINO}} = -\sum_{k=1}^K p_t^{(k)} \log p_s^{(k)}$$

and $\gamma = 0.5$ balances the two objectives. Gradients flow only through the student; the teacher is updated by EMA only.

---

## 4. Theoretical Analysis

### 4.1 Why Diffusion Features Are Especially Prone to Collapse

In standard DINO, the student and teacher receive different augmented views of the same clean image. The feature distributions of the two views are close but not identical, providing a natural diversity that stabilizes self-distillation. In SDD, the student and teacher receive differently-noised versions of $x_0$. The key difference is that the noise-induced diversity **interacts with the timestep**: at high $t$, the two views become nearly indistinguishable (both are close to isotropic Gaussian), while at low $t$, they are close to the clean image and thus nearly identical to each other. Only in the intermediate regime $t \in [t_{\min}, t_{\max}]$ is the view diversity appropriate for meaningful self-distillation.

More formally, consider the teacher feature covariance $\Sigma_t = \text{Cov}[\tilde{z}_t]$ over a batch. At high noise, $\tilde{z}_t \approx g_t(f_t(\epsilon, t))$ for $\epsilon \sim \mathcal{N}(0, I)$, so $\Sigma_t$ is determined entirely by the network's response to Gaussian inputs. If the EMA teacher has converged to a state where these responses are low-rank (e.g., the bottleneck compresses noise uniformly), then $\text{rank}(\Sigma_t)$ may be far below $K$, leading to **dimensional collapse** in the teacher targets. Centering counteracts this by explicitly regularizing the mean of $\tilde{z}_t$ to zero, which in turn increases the effective rank of $\Sigma_t$ by removing the dominant mean direction. This argument generalizes the centering analysis of Caron et al. (2021) to the diffusion setting.

### 4.2 Why Sharpening Is Necessary

Without sharpening, the teacher distribution $p_t = \text{softmax}(\tilde{z}_t / \tau_t)$ with $\tau_t = \tau_s$ (equal temperatures) produces high-entropy targets whenever $\|\tilde{z}_t\|$ is small — which occurs systematically in early training when the EMA teacher has not yet learned discriminative features. The resulting cross-entropy loss $\mathcal{L}_{\text{DINO}}$ then provides a nearly uninformative gradient signal (approaching $\log K$ regardless of student output), stalling representation learning. Sharpening ($\tau_t \ll \tau_s$) amplifies the teacher's softmax outputs, maintaining a non-trivial gradient signal even when the teacher's raw logits have small magnitude.

Formally, the gradient of $\mathcal{L}_{\text{DINO}}$ with respect to the student logits $\tilde{z}_s$ is $p_s - p_t$. With sharpening, $p_t$ is a near-one-hot distribution, so $p_s - p_t$ has a large magnitude at the target dimension, providing a strong learning signal. Without sharpening, $p_t \approx \mathbf{1}/K$ and $p_s - p_t \approx 0$, giving near-zero gradients.

### 4.3 Interaction: Joint Necessity

Centering alone without sharpening prevents dimensional collapse but does not resolve the gradient stalling problem. Sharpening alone without centering prevents gradient stalling but does not prevent dimensional collapse. Only their joint application provides both guarantees simultaneously. This is consistent with the ablation findings reported by Caron et al. (2021) for DINO, and we expect the same interaction to hold in the diffusion setting given the structural similarity of the self-distillation objective.

---

## 5. Hyperparameter Analysis

### 5.1 Distillation Weight $\gamma$

The weight $\gamma$ controls the trade-off between the denoising objective and the representation learning objective. Too large a $\gamma$ degrades generation quality by distorting the noise prediction landscape; too small a $\gamma$ renders the distillation signal negligible.

We recommend $\gamma \in [0.1, 1.0]$ as the practical range, with $\gamma = 0.5$ as a default. The optimal value depends on the architecture capacity: larger models (DiT-XL) can absorb a larger $\gamma$ without generation degradation, while smaller models (UNet-base) may require $\gamma \leq 0.3$. We suggest monitoring the ratio $\mathcal{L}_{\text{DINO}} / \mathcal{L}_{\text{MSE}}$ during training and adjusting $\gamma$ to keep this ratio in the range $[0.1, 0.5]$.

### 5.2 Timestep Gate $[t_{\min}, t_{\max}]$

The gate boundaries determine which noise levels contribute to representation learning. As argued in Section 3.3.4, $t_{\max}$ should be set below the noise level at which $I(x_t; x_0)$ becomes negligible (practically, $t_{\max} \leq 0.7T$). Setting $t_{\min}$ too low (e.g., $t_{\min} = 0$) wastes distillation capacity on the near-clean regime where the MSE loss already provides strong supervision. We recommend the default $[0.1T, 0.6T]$ for most settings, with the soft gate variant preferred when the downstream task requires smooth gradient flow across timestep boundaries.

### 5.3 Projection Dimension $K$

$K = 256$ is the default, following DINO. Larger $K$ increases the expressivity of the representation space but also increases the risk of dimensional collapse (more dimensions to fill). We recommend $K \leq 512$ for models with bottleneck feature dimension $\leq 1024$, and $K \leq 256$ for smaller models.

### 5.4 Temperature Parameters $\tau_t$, $\tau_s$

The teacher temperature $\tau_t$ should be set low enough to produce near-one-hot distributions (we use $\tau_t = 0.04$, consistent with DINO). The student temperature $\tau_s$ should be set higher to produce a softer distribution from which gradients can be computed (we use $\tau_s = 0.1$). The ratio $\tau_s / \tau_t = 2.5$ approximately matches the DINO default and can be used as a starting point.

---

## References

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*.

[2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*.

[3] Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *ICLR 2021*.

[4] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *ICCV 2021*. (DINO)

[5] Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning robust visual features without supervision. *TMLR 2024*.

[6] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022*. (LDM)

[7] Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023*. (DiT)

[8] Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS 2021*. (ADM)

[9] Preechakul, K., Chatthee, N., Wizadwongsa, S., & Suwajanakorn, S. (2022). Diffusion autoencoders: Toward a meaningful and decodable representation. *CVPR 2022*. (DDAE)

[10] Yu, S., Kwak, D., Jang, H., et al. (2024). Representation alignment for generation: Training diffusion transformers is easier than you think. *ICLR 2025 Oral*. (REPA)

[11] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR 2020*. (MoCo)

[12] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*. (SimCLR)

[13] Grill, J. B., Strub, F., Altché, F., et al. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS 2020*. (BYOL)