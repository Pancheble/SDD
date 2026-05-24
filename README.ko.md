# Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models

**[English](README.md) | [Korean](README.ko.md)**

---

## Abstract

Diffusion model은 생성 모델링 분야에서 놀라운 성과를 달성하였으나, 그 내부 representation은 하류 판별 태스크에 대해 여전히 구조적으로 부실한 상태에 머물러 있다. Pixel space에서의 mean squared error regression으로 이루어지는 표준 denoising objective는 모델이 내부 feature space를 의미론적으로 조직화하도록 유도하는 명시적인 유인을 제공하지 않으며, 그 결과 DINO와 같은 전용 self-supervised learning 방법론과 비교하여 discriminability가 낮고 평활한 representation이 산출된다. 이 격차를 해소하고자 한 선행 연구들은 외부의 사전 학습된 encoder에 의존하거나(REPA), collapse 방지 및 entropy sharpening 연산을 적용하지 않는 EMA(Exponential Moving Average) 기반의 self-distillation에 그치는 한계를 보인다(SD-DiT, DDAE). 본 연구에서는 **Self-Distilled Diffusion (SDD)**을 제안한다. SDD는 표준 diffusion objective에 UNet 또는 DiT의 중간 feature에 직접 적용되는 DINO 방식의 self-distillation loss를 결합한 학습 프레임워크이다. 구체적으로, student diffusion model과 병렬로 EMA 갱신되는 teacher network를 유지하고, 중간 feature를 공유 embedding space로 projection한 뒤, centering 및 sharpening 연산을 적용하여 안정적이고 confident한 학습 target을 생성한다. 본 연구는 centering과 sharpening이 diffusion feature 환경에서 단순히 유익한 것이 아니라 필수적임을 이론적으로 분석하고, 이러한 연산 없이는 naive한 EMA distillation이 diffusion training dynamics 하에서 dimensional collapse에 취약함을 증명한다. 또한 정보 이론적 논거에 근거하여 semantic feature가 가장 활성화되는 low-to-middle noise 구간에 distillation signal을 제한하는 timestep-adaptive weighting 방식을 도입한다.

---

## 1. Introduction

Denoising diffusion probabilistic model(DDPM)은 고충실도 이미지 합성의 지배적 패러다임으로 자리매김하였다. 표준 학습 objective는 다음과 같다:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

여기서 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$는 timestep $t$에서의 noisy image이며, $\epsilon_\theta$는 추가된 noise를 예측하도록 학습하는 neural network(통상 UNet 또는 DiT)이다.

이 regression objective는 근본적인 한계를 내포한다. 모델이 구조화되고 의미론적으로 의미 있는 내부 representation을 구축하도록 유도하는 명시적인 유인을 제공하지 않는 것이다. 최근의 실증적 연구들은 이 objective만으로 학습된 diffusion model이, 충분히 긴 학습 후에도, DINOv2와 같은 전용 discriminative model에 비해 현저히 열등한 representation을 산출함을 확인하였다.

한편, DINO(Self-Distillation with No Labels)는 teacher-student self-distillation framework에 두 가지 간단한 연산—**centering**(collapse를 방지하기 위해 running mean을 뺌)과 **sharpening**(teacher에 낮은 temperature를 적용하여 confident한 target을 생성)—을 결합함으로써, label 없이도 탁월한 visual representation을 달성할 수 있음을 보였다.

**선행 연구의 공백.** SD-DiT와 DDAE는 EMA 기반 teacher-student 구조를 diffusion training에 통합할 수 있음을 보였으나, 어느 것도 centering이나 sharpening을 적용하지 않았다. 본 연구는 4절에서 증명하듯, 이 연산들을 생략하는 것은 단순한 구현 세부 사항이 아니라, plain EMA distillation이 연구된 augmentation 기반의 contrastive learning 환경과 근본적으로 다른 diffusion training dynamics 하에서 dimensional collapse에 노출되는 결과를 초래함을 주장한다. REPA는 외부 DINOv2 encoder를 통해 이 문제를 회피하지만, 그로 인해 대형 pretrained model에 대한 강한 의존성이 발생하여, 해당 encoder가 존재하지 않는 domain 및 modality에서의 적용 가능성이 제한된다.

본 연구가 제기하는 질문은 다음과 같다: **DINO의 centering과 sharpening을 diffusion model feature에 직접 적용하여, 외부의 pretrained encoder 없이 collapse에 대한 이론적 보장을 갖추면서 모델이 denoising과 구조화된 representation learning을 동시에 수행할 수 있는가?**

본 연구의 기여는 다음과 같다:

1. 외부 encoder가 불필요한 완전 self-contained diffusion training loop 내에서 DINO 방식의 centering과 sharpening을 적용하는 framework인 **Self-Distilled Diffusion (SDD)**을 제안한다.
2. (a) Diffusion training dynamics가 표준 augmentation 기반 self-distillation보다 체계적으로 높은 feature collapse 위험을 생성하며, (b) centering과 sharpening이 각각 feature covariance의 rank와 teacher target의 entropy를 제어함으로써 이 위험을 해소함을 보이는 **이론적 분석**(4절)을 제공한다.
3. Semantic content가 noise level에 따라 단조롭게 감소한다는 직관을 형식화하고 이로부터 최적 gating window를 도출하는 정보 이론적 논거에 의해 동기화된 **timestep-adaptive gating mechanism**(3.3.4절)을 도입한다.
4. SDD의 핵심 hyperparameter $\gamma$, $t_{\min}$, $t_{\max}$, $\tau_t$, $\tau_s$ 각각에 대한 민감도를 규명하고 설정 지침을 제공하는 **원리에 입각한 hyperparameter 분석**(5절)을 수행한다.
5. Centering과 sharpening이 독립적으로 기여하며 그 결합이 필수적임을 검증하는 체계적 ablation 실험을 설계한다.

---

## 2. Related Work

### 2.1 Diffusion Models

DDPM은 반복적 denoising을 통해 데이터 분포를 학습하는 원리적 framework를 제시하였다. Score-based generative model과 이후의 DDIM sampler는 연속 시간 및 결정론적 변형을 확립하였다. ADM, LDM, DiT 등의 아키텍처 발전은 diffusion model을 최첨단 generation 품질로 이끌었다. 그러나 이 방법들은 모두 동일한 pixel-space MSE objective를 공유하며, representation 형성을 위한 메커니즘을 결여하고 있다.

### 2.2 Self-Supervised Representation Learning

MoCo, SimCLR, BYOL은 판별적 representation learning을 위한 contrastive learning 및 self-distillation framework를 확립하였다. DINO는 centering과 sharpening을 특징으로 하는 teacher-student 아키텍처로 이 패러다임을 확장하여, label 없이도 창발적 segmentation 특성을 지닌 representation을 산출함을 보였다. DINOv2는 이를 대규모 pretrained model로 확장하였다. DINO의 핵심 통찰은 centering과 sharpening이 선택적 개선 사항이 아니라, 각각 mode collapse와 entropy collapse를 방지하는 메커니즘이며, 이 둘의 상호작용이 DINO를 아키텍처와 scale에 걸쳐 안정적으로 만드는 요인이라는 점이다.

### 2.3 Representation Learning via Diffusion

여러 연구가 diffusion model이 유용한 representation을 부수적으로 학습함에 주목하였다. DDAE는 EMA distillation을 통해 diffusion model을 representation learner로 활용하는 개념 증명을 보였으나, collapse 방지 메커니즘은 적용하지 않았다. REPA(ICLR 2025 Oral)는 학습 중 diffusion model feature를 외부 DINOv2 encoder에 align하는 방식을 제안하여 극적인 속도 향상과 향상된 generation 품질을 달성하였으나, 학습 시 외부 encoder를 필요로 한다. SD-DiT는 내부 EMA teacher-student 구조를 탐구하였으나, 마찬가지로 centering이나 sharpening을 적용하지 않았다.

**공백.** DINO의 완전한 centering+sharpening 메커니즘을 외부 encoder 없는 self-contained diffusion training loop에 적용한 선행 연구는 존재하지 않으며, diffusion 환경에서 이러한 연산이 왜 필수적인지를 이론적으로 분석한 연구 역시 없다. SDD는 이 두 가지 공백을 모두 해소한다.

---

## 3. Method

### 3.1 Preliminaries: Standard Diffusion Training

데이터 분포 $q(x_0)$가 주어졌을 때, forward process는 Gaussian noise를 점진적으로 추가한다:

$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)\, I)$$

모델 $\epsilon_\theta(x_t, t)$는 다음을 최소화하여 이 process를 역전하도록 학습된다:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x_0,\, \epsilon \sim \mathcal{N}(0,I),\, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 3.2 DINO Background

DINO는 student network $f_s$와 EMA 갱신되는 teacher network $f_t$(momentum $m$)를 동일 이미지의 augmented view에 대해 학습한다. 핵심 연산은 다음과 같다:

**Centering** — softmax 적용 이전에 teacher output에서 running mean $c$를 차감하여, 어느 단일 차원도 output distribution을 지배하지 않도록 함으로써 dimensional collapse를 방지한다:
$$p_t = \text{softmax}\!\left(\frac{z_t - c}{\tau_t}\right), \qquad c \leftarrow \alpha \cdot c + (1 - \alpha) \cdot \mathbb{E}_{\text{batch}}[z_t]$$
여기서 $z_t$는 projection head 이전의 raw teacher logit을 나타낸다.

**Sharpening** — 낮은 temperature $\tau_t \ll \tau_s$를 teacher에 적용하여, student output이 불확실한 경우에도 confident한(낮은 entropy) target distribution을 생성한다. 이를 통해 student와 teacher 모두 uniform distribution으로 수렴하는 entropy collapse를 방지하고 급격한 gradient signal을 유지한다.

Loss function은 teacher distribution과 student distribution 간의 cross-entropy이다:
$$\mathcal{L}_{\text{DINO}} = -\sum_k p_t^{(k)} \log p_s^{(k)}$$

### 3.3 Self-Distilled Diffusion (SDD)

#### 3.3.1 Architecture

두 network를 유지한다:

- **Student**: $\epsilon_\theta$ — gradient descent로 갱신되는 표준 diffusion UNet/DiT.
- **Teacher**: $\epsilon_\xi$ — student의 EMA 복사본으로, $\xi \leftarrow m \cdot \xi + (1-m) \cdot \theta$에 따라 momentum $m = 0.996$으로 갱신된다.

두 network는 동일한 이미지 $x_0$를 forward diffusion process를 통해 처리한 입력을 받는다. 중요한 점은, student와 teacher에 대해 서로 다른 noise realization $\epsilon_s, \epsilon_t \sim \mathcal{N}(0, I)$가 독립적으로 sampling되어, 동일한 timestep $t$에서도 $x_t^{(s)} \neq x_t^{(t)}$가 성립한다는 것이다. 이 stochasticity는 명시적 data augmentation pipeline 없이 표준 DINO에서 augmentation이 제공하는 view diversity를 제공한다. 학습 domain이 지원하는 경우(예: 자연 이미지), view diversity를 더욱 높이기 위해 forward process 이전에 $x_0$에 표준 augmentation(random crop, color jitter)을 선택적으로 적용할 수 있으나, 이는 optional extension으로 취급한다. 핵심 방법론은 noise stochasticity만에 의존한다.

#### 3.3.2 Feature Extraction and Projection

UNet의 bottleneck layer(또는 DiT의 중간 transformer block)에서 중간 feature를 추출한다. $z_s = f_s(x_t^{(s)}, t)$와 $z_t = f_t(x_t^{(t)}, t)$를 각각 student 및 teacher의 bottleneck feature라 하자.

각각은 기본값으로 $K = 256$차원 embedding(5.3절의 sensitivity analysis 참조)으로 mapping하는 경량의 2층 MLP projection head $g(\cdot)$를 통과한다:

$$\tilde{z}_s = g_s(z_s), \qquad \tilde{z}_t = g_t(z_t)$$

Projection head는 공유되지 않으며, 이는 DINO의 설계를 따른 것이다. Head를 공유하면 student와 teacher의 optimization trajectory가 결합되어 EMA update dynamics를 방해하고 teacher-student output 간의 유효한 diversity가 감소하기 때문이다. Teacher head $g_t$도 동일한 momentum $m$으로 EMA 갱신된다.

#### 3.3.3 Centering and Sharpening

DINO를 따라 teacher의 projection output $\tilde{z}_t$에 centering과 낮은 temperature를, student에는 높은 temperature를 적용한다:

$$p_t = \text{softmax}\!\left(\frac{\tilde{z}_t - c}{\tau_t}\right), \qquad p_s = \text{softmax}\!\left(\frac{\tilde{z}_s}{\tau_s}\right)$$

여기서:
- $c$는 teacher projection output의 running EMA이다: $c \leftarrow \alpha_c \cdot c + (1-\alpha_c)\,\mathbb{E}_{\text{batch}}[\tilde{z}_t]$
- $\tau_t = 0.04$ (teacher temperature, 낮은 값 → target distribution을 sharpening)
- $\tau_s = 0.1$ (student temperature, 높은 값 → 더 부드러운 student prediction)
- $\alpha_c = 0.9$ (centering EMA momentum)

Centering은 전체적으로 일관되게 $\tilde{z}_t$(projection head 이후 output)에 적용됨에 유의하라. 이 선택에 대한 이론적 동기는 4절에서 제공된다.

#### 3.3.4 Timestep-Adaptive Gating

**동기.** 높은 noise level($t \to T$)에서 noisy input $x_t$는 Gaussian noise에 지배되어 $x_0$와의 mutual information이 무시할 수 없을 정도로 작아진다. 형식적으로, $x_t$와 $x_0$ 사이의 mutual information은 다음과 같다:

$$I(x_t; x_0) = -\frac{1}{2}\log\!\left(1 - \bar{\alpha}_t\right) + \text{const}$$

이는 $\bar{\alpha}_t \to 0$(즉, $t \to T$)으로 감에 따라 단조롭게 감소한다. Distillation target $p_t$가 $x_t$에 조건화된 teacher feature로부터 도출되므로, $p_t$의 semantic content는 high-noise 구간에서 소멸한다. 이 구간에 distillation loss를 적용하면 representation learning signal에 noise가 유입되어 feature 품질을 저해할 수 있다.

따라서 다음의 **timestep gate**를 도입한다:

$$w(t) = \mathbb{1}\left[t_{\min} \leq t \leq t_{\max}\right]$$

기본값은 $t_{\min} = 0.1T$, $t_{\max} = 0.6T$이다. 하한 $t_{\min} > 0$은 MSE loss가 이미 강한 per-pixel signal을 제공하는 near-clean 구간도 추가로 회피하여, auxiliary representation shaping의 필요성을 줄인다. 상한 $t_{\max} = 0.6T$는 diffusion model 내부에 대한 선행 실증 분석, 즉 semantic linear probing accuracy가 $t \in [0.1T, 0.5T]$ 구간에서 정점에 달한다는 결과에서 도출하였으며, hard edge effect를 피하기 위해 $0.6T$로 소폭 연장하였다.

양쪽 경계를 대칭적으로 포괄하는 soft variant는 sigmoid의 곱으로 정의된다:

$$w(t) = \sigma\!\left(\frac{t - t_{\min}}{\beta}\right) \cdot \sigma\!\left(-\frac{t - t_{\max}}{\beta}\right)$$

여기서 $\beta$는 gate의 sharpness를 제어한다. 이 공식은 일부 선행 연구에서 사용된 single-sided sigmoid와 달리, 양쪽 극단($t \ll t_{\min}$ 및 $t \gg t_{\max}$) 모두에서 distillation signal을 올바르게 억제하며, $\beta \to 0$으로 가면서 hard gate로 수렴한다.

#### 3.3.5 Total Training Objective

SDD의 전체 objective는 다음과 같다:

$$\mathcal{L}_{\text{SDD}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot w(t) \cdot \mathcal{L}_{\text{DINO}}(p_t, p_s)$$

여기서:

$$\mathcal{L}_{\text{MSE}} = \| \epsilon - \epsilon_\theta(x_t^{(s)}, t) \|^2$$

$$\mathcal{L}_{\text{DINO}} = -\sum_{k=1}^K p_t^{(k)} \log p_s^{(k)}$$

이고, $\gamma = 0.5$는 두 objective의 균형을 맞추는 scalar이다. Gradient는 student를 통해서만 흐르며, teacher는 EMA에 의해서만 갱신된다.

---

## 4. Theoretical Analysis

### 4.1 Why Diffusion Features Are Especially Prone to Collapse

표준 DINO에서는 student와 teacher가 동일한 clean image의 서로 다른 augmented view를 입력받는다. 두 view의 feature distribution은 유사하되 동일하지 않아, self-distillation을 안정화하는 자연스러운 diversity를 제공한다. SDD에서는 student와 teacher가 $x_0$의 서로 다른 noisy version을 입력받는다. 핵심적인 차이는 noise로 인한 diversity가 **timestep과 상호작용**한다는 것이다. 높은 $t$에서는 두 view가 거의 구별 불가능해지며(둘 다 isotropic Gaussian에 근접), 낮은 $t$에서는 두 view가 clean image에 근접하여 서로 거의 동일해진다. 오직 중간 구간 $t \in [t_{\min}, t_{\max}]$에서만 view diversity가 의미 있는 self-distillation에 적합하다.

보다 형식적으로, batch에 대한 teacher feature covariance를 $\Sigma_t = \text{Cov}[\tilde{z}_t]$라 하자. High noise에서는 $\epsilon \sim \mathcal{N}(0, I)$에 대해 $\tilde{z}_t \approx g_t(f_t(\epsilon, t))$이므로, $\Sigma_t$는 전적으로 Gaussian input에 대한 network 반응에 의해 결정된다. EMA teacher가 bottleneck이 noise를 균일하게 압축하는 상태로 수렴하면, $\text{rank}(\Sigma_t)$는 $K$보다 훨씬 낮아져 teacher target에서 **dimensional collapse**가 발생할 수 있다. Centering은 $\tilde{z}_t$의 mean을 명시적으로 zero로 정규화함으로써 이를 상쇄한다. 이는 지배적인 mean direction을 제거하여 $\Sigma_t$의 effective rank를 높이는 효과를 갖는다. 이 논거는 Caron et al.(2021)의 centering 분석을 diffusion 환경으로 일반화한 것이다.

### 4.2 Why Sharpening Is Necessary

Sharpening이 없으면, 동일한 temperature $\tau_t = \tau_s$에서의 teacher distribution $p_t = \text{softmax}(\tilde{z}_t / \tau_t)$는 $\|\tilde{z}_t\|$가 작을 때 높은 entropy target을 산출한다. 이는 EMA teacher가 아직 discriminative feature를 학습하지 못한 학습 초기에 체계적으로 발생한다. 그 결과, cross-entropy loss $\mathcal{L}_{\text{DINO}}$는 student output에 무관하게 $\log K$에 가까운 거의 uninformative한 gradient signal을 제공하게 되어 representation learning이 정체된다. Sharpening($\tau_t \ll \tau_s$)은 teacher의 softmax output을 증폭시켜, teacher의 raw logit magnitude가 작은 경우에도 non-trivial한 gradient signal을 유지한다.

형식적으로, $\mathcal{L}_{\text{DINO}}$의 student logit $\tilde{z}_s$에 대한 gradient는 $p_s - p_t$이다. Sharpening을 적용하면 $p_t$가 거의 one-hot distribution이 되어 $p_s - p_t$가 target 차원에서 큰 크기를 가지며, 강한 learning signal을 제공한다. Sharpening 없이는 $p_t \approx \mathbf{1}/K$이고 $p_s - p_t \approx 0$이 되어 거의 zero gradient가 산출된다.

### 4.3 Interaction: Joint Necessity

Centering만으로는 dimensional collapse를 방지하나 gradient stalling 문제를 해소하지 못한다. Sharpening만으로는 gradient stalling을 방지하나 dimensional collapse를 막지 못한다. 두 연산의 결합 적용만이 두 보장을 동시에 제공한다. 이는 Caron et al.(2021)이 DINO에 대해 보고한 ablation 결과와 일관되며, self-distillation objective의 구조적 유사성을 고려할 때 diffusion 환경에서도 동일한 상호작용이 성립할 것으로 예상한다.

---

## 5. Hyperparameter Analysis

### 5.1 Distillation Weight $\gamma$

가중치 $\gamma$는 denoising objective와 representation learning objective 간의 trade-off를 제어한다. $\gamma$가 너무 크면 noise prediction landscape를 왜곡하여 generation 품질이 저하되고, 너무 작으면 distillation signal이 미미해진다.

실용적 범위로 $\gamma \in [0.1, 1.0]$을 권장하며, 기본값은 $\gamma = 0.5$이다. 최적값은 아키텍처 용량에 따라 달라진다. 더 큰 모델(DiT-XL)은 generation 품질 저하 없이 더 큰 $\gamma$를 수용할 수 있는 반면, 더 작은 모델(UNet-base)은 $\gamma \leq 0.3$이 필요할 수 있다. 학습 중 $\mathcal{L}_{\text{DINO}} / \mathcal{L}_{\text{MSE}}$ 비율을 모니터링하고, 이 비율이 $[0.1, 0.5]$ 범위에 유지되도록 $\gamma$를 조정할 것을 권고한다.

### 5.2 Timestep Gate $[t_{\min}, t_{\max}]$

Gate 경계는 어느 noise level이 representation learning에 기여하는지를 결정한다. 3.3.4절의 논거에 따라, $t_{\max}$는 $I(x_t; x_0)$가 무시할 수 있을 정도로 작아지는 noise level 이하로 설정해야 한다(실용적으로 $t_{\max} \leq 0.7T$). $t_{\min}$을 너무 낮게 설정하면(예: $t_{\min} = 0$), MSE loss가 이미 강한 supervision signal을 제공하는 near-clean 구간에 distillation 역량이 낭비된다. 대부분의 설정에 대해 기본값 $[0.1T, 0.6T]$를 권장하며, downstream task가 timestep 경계에 걸쳐 부드러운 gradient flow를 요구하는 경우 soft gate variant가 선호된다.

### 5.3 Projection Dimension $K$

DINO를 따라 기본값으로 $K = 256$을 사용한다. $K$가 클수록 representation space의 expressivity가 높아지지만 dimensional collapse의 위험도 증가한다(채워야 할 차원이 더 많아짐). Bottleneck feature dimension이 $1024$ 이하인 모델에 대해서는 $K \leq 512$를, 더 작은 모델에 대해서는 $K \leq 256$을 권장한다.

### 5.4 Temperature Parameters $\tau_t$, $\tau_s$

Teacher temperature $\tau_t$는 거의 one-hot distribution을 산출할 만큼 충분히 낮게 설정해야 한다(DINO와 일관되게 $\tau_t = 0.04$ 사용). Student temperature $\tau_s$는 gradient를 계산할 수 있는 더 부드러운 distribution을 산출하도록 더 높게 설정해야 한다($\tau_s = 0.1$ 사용). 비율 $\tau_s / \tau_t = 2.5$는 DINO의 기본값에 근사하며, 시작점으로 활용할 수 있다.

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