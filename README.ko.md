# Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models

**[English](README.md) | [Korean](README.ko.md)**

---

## Abstract

Diffusion 모델은 생성 모델링 분야에서 괄목할 만한 성과를 거두었으나, 그 내부 표현은 하위 판별 태스크(downstream discriminative task)에 활용하기에 충분히 구조화되어 있지 않다. 표준적인 노이즈 제거 목적 함수, 즉 픽셀 공간에서의 평균 제곱 오차(MSE) 회귀는 매끄럽고 판별력이 낮은 특징을 생성하며, DINO와 같은 전용 자기 지도 표현 학습기에 비해 성능이 열세하다. 본 연구에서는 **Self-Distilled Diffusion (SDD)**을 제안한다. SDD는 표준 diffusion 목적 함수에 DINO 방식의 자기 증류(self-distillation) 손실을 UNet 중간 특징에 직접 적용하는 훈련 프레임워크이다. 구체적으로, EMA로 갱신되는 교사(teacher) 네트워크를 학생(student) diffusion 모델과 함께 유지하고, 중간 특징을 공유 임베딩 공간에 투영한 뒤, centering 및 sharpening 연산을 적용하여 안정적이고 확신도 높은 훈련 목표를 생성한다. REPA와 같은 선행 연구와 달리, 본 방법은 외부 사전 학습 인코더를 필요로 하지 않으며 완전히 자기 완결적(self-contained)이다. 또한, 의미론적 특징이 가장 활성화되는 낮은-중간 노이즈 구간으로 증류 신호를 제한하는 타임스텝 적응형 가중치 기법을 추가로 제안한다.

---

## 1. Introduction

노이즈 제거 확산 확률 모델(Denoising Diffusion Probabilistic Models, DDPMs)은 고품질 이미지 합성의 지배적인 패러다임으로 자리잡았다. 표준 훈련 목적 함수는 다음과 같다.

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

여기서 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$은 타임스텝 $t$에서의 노이즈가 추가된 이미지이며, $\epsilon_\theta$는 추가된 노이즈를 예측하도록 학습되는 신경망(주로 UNet 또는 DiT)이다.

이 회귀 목적 함수는 근본적인 한계를 지닌다. 모델이 구조화되고 의미론적으로 풍부한 내부 표현을 형성하도록 유도하는 명시적인 장치가 존재하지 않는다는 점이다. 최근의 실증 연구들은 이 목적 함수만으로 훈련된 diffusion 모델이, 장시간 훈련 후에도 DINOv2와 같은 전용 판별 모델에 비해 현저히 열등한 표현을 생성함을 확인한 바 있다.

한편, DINO(Self-Distillation with No Labels)는 교사-학생 자기 증류 프레임워크에 두 가지 간단한 연산, 즉 **centering**(붕괴(collapse) 방지를 위해 누적 평균을 차감하는 연산)과 **sharpening**(확신도 높은 목표 분포 생성을 위해 교사에 낮은 온도를 적용하는 연산)을 결합함으로써, 레이블 없이도 탁월한 시각적 표현을 획득할 수 있음을 입증하였다.

이에 본 연구에서는 다음과 같은 자연스러운 물음을 제기한다. **DINO의 centering과 sharpening을 diffusion 모델의 특징에 직접 적용함으로써, 외부 사전 학습 인코더에 의존하지 않고도 모델이 노이즈 제거와 구조화된 표현 학습을 동시에 수행할 수 있는가?**

본 연구의 기여는 다음과 같다.

1. 외부 인코더 없이 완전히 자기 완결적인 diffusion 훈련 루프 내에서 DINO 방식의 centering과 sharpening을 적용하는 프레임워크인 **Self-Distilled Diffusion (SDD)**을 제안한다.
2. 의미론적 특징이 가장 활성화되는 노이즈 구간에서만 증류 손실을 선택적으로 적용하는 **타임스텝 적응형 게이팅 메커니즘**을 도입하여, 단순 적용 시 발생하는 특징-타임스텝 충돌 문제를 해소한다.
3. centering과 sharpening이 각각 독립적으로 표현 품질에 기여하며, 두 연산의 결합이 SD-DiT 및 DDAE 등 선행 연구에서 사용된 centering·sharpening 없는 EMA 기반 증류보다 우수함을 검증하기 위한 체계적인 ablation 실험을 설계한다.

---

## 2. Related Work

### 2.1 Diffusion Models

DDPM은 반복적인 노이즈 제거를 통해 데이터 분포를 학습하는 원칙적인 프레임워크를 제시하였다. 이후 스코어 기반 생성 모델과 DDIM 샘플러는 연속 시간 및 결정론적 변형을 확립하였으며, ADM, LDM, DiT 등의 아키텍처 발전을 통해 diffusion 모델은 최고 수준의 생성 품질을 달성하였다. 그러나 이들 방법은 모두 동일한 픽셀 공간 MSE 목적 함수를 공유하며, 표현 형성을 위한 별도의 메커니즘을 갖추지 않고 있다.

### 2.2 Self-Supervised Representation Learning

MoCo, SimCLR, BYOL은 판별적 표현 학습을 위한 대조적·자기 증류 프레임워크를 확립하였다. DINO는 centering과 sharpening을 갖춘 교사-학생 아키텍처를 통해 이 패러다임을 확장하여, 레이블 없이도 자발적인 분할(segmentation) 특성을 보이는 표현을 생성함을 보였다. DINOv2는 이를 대규모 사전 학습 모델로 확장하였다. 본 프로젝트는 centering과 sharpening 메커니즘을 생성적 diffusion 훈련 루프에 이식함으로써 이 공백을 채우는 것을 목표로 한다.

### 2.3 Representation Learning via Diffusion

여러 연구에서 diffusion 모델이 부수적으로 유용한 표현을 학습함이 관찰되었다. DDAE는 EMA 증류를 통한 표현 학습에 diffusion 모델을 활용하는 방법을 탐색하였다. REPA(ICLR 2025 Oral)는 훈련 과정에서 diffusion 모델의 특징을 외부 DINOv2 인코더에 정렬함으로써 훈련 속도와 생성 품질을 크게 향상시켰다. SRA와 SD-DiT는 diffusion 훈련 내부에서 EMA 교사-학생 구조를 탐구하였으나, centering과 sharpening을 적용하지는 않았다. 본 프로젝트는 외부 인코더 없는 자기 완결적 diffusion 훈련 루프 내에서 DINO의 centering과 sharpening 메커니즘을 완전히 적용함으로써 이 공백을 해소하고자 한다.

---

## 3. Method

### 3.1 Preliminaries: Standard Diffusion Training

데이터 분포 $q(x_0)$가 주어졌을 때, 순방향 과정은 다음과 같이 가우시안 노이즈를 점진적으로 추가한다.

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

모델 $\epsilon_\theta(x_t, t)$는 다음 목적 함수를 최소화함으로써 이 과정을 역전하도록 훈련된다.

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 3.2 DINO Background

DINO는 동일 이미지의 증강된 뷰(augmented view)에 대해 학생 네트워크 $f_s$와 EMA로 갱신되는 교사 네트워크 $f_t$(모멘텀 $m$)를 함께 훈련한다. 핵심 연산은 다음과 같다.

**Centering** — 붕괴(trivial solution collapse) 방지를 위해 softmax 적용 전 교사 출력에서 누적 평균 $c$를 차감한다.
$$p_t = \text{softmax}\left(\frac{z_t - c}{\tau_t}\right), \quad c \leftarrow \alpha \cdot c + (1 - \alpha) \cdot \mathbb{E}_{\text{batch}}[z_t]$$

**Sharpening** — 학생 출력이 불확실한 경우에도 확신도 높은(저엔트로피) 목표 분포를 생성하기 위해, 교사에 낮은 온도 $\tau_t \ll \tau_s$를 적용한다.

손실 함수는 교사 분포와 학생 분포 사이의 교차 엔트로피로 정의된다.

$$\mathcal{L}_{\text{DINO}} = -\sum_k p_t^{(k)} \log p_s^{(k)}$$

### 3.3 Self-Distilled Diffusion (SDD)

#### 3.3.1 Architecture

본 방법은 두 개의 네트워크를 유지한다.

- **학생(Student)**: $\epsilon_\theta$ — 경사 하강법(gradient descent)으로 갱신되는 표준 diffusion UNet/DiT.
- **교사(Teacher)**: $\epsilon_\xi$ — 학생의 EMA 복사본으로, $\xi \leftarrow m \cdot \xi + (1-m) \cdot \theta$ (모멘텀 $m = 0.996$)으로 갱신된다.

두 네트워크는 동일한 원본 이미지 $x_0$에 대해 독립적으로 증강된 뷰(예: 랜덤 크롭, 색상 지터(color jitter))를 각각 입력받으며, 각 뷰는 순방향 확산 과정을 거쳐 해당 네트워크의 노이즈 입력 $x_t$로 변환된다.

#### 3.3.2 Feature Extraction and Projection

UNet의 병목(bottleneck) 레이어(또는 DiT의 중간 트랜스포머 블록)에서 중간 특징을 추출한다. $z_s = f_s(x_t, t)$와 $z_t = f_t(x_t, t)$를 각각 학생과 교사의 병목 특징이라 하자.

각 특징은 $K$차원 임베딩 공간(기본값 $K = 256$)으로 매핑하는 경량 2층 MLP 투영 헤드 $g(\cdot)$를 통과한다.

$$\tilde{z}_s = g_s(z_s), \quad \tilde{z}_t = g_t(z_t)$$

투영 헤드는 공유되지 않으며, 교사 헤드 $g_t$ 역시 EMA로 갱신된다.

#### 3.3.3 Centering and Sharpening

DINO의 방식을 그대로 따라, 교사 로짓에는 centering과 낮은 온도를, 학생에는 높은 온도를 적용한다.

$$p_t = \text{softmax}\left(\frac{\tilde{z}_t - c}{\tau_t}\right), \quad p_s = \text{softmax}\left(\frac{\tilde{z}_s}{\tau_s}\right)$$

각 기호의 정의는 다음과 같다.
- $c$: 배치마다 교사 출력의 누적 EMA로 갱신. $c \leftarrow \alpha_c \cdot c + (1-\alpha_c) \mathbb{E}[\tilde{z}_t]$
- $\tau_t = 0.04$: 교사 온도 (낮을수록 목표 분포가 날카로워짐)
- $\tau_s = 0.1$: 학생 온도 (높을수록 학생 예측이 부드러워짐)
- $\alpha_c = 0.9$: centering EMA 모멘텀

#### 3.3.4 Timestep-Adaptive Gating

모든 타임스텝에 걸쳐 증류 손실을 단순히 적용하는 방식은 문제가 있다. 높은 노이즈 수준($t \to T$)에서는 입력이 거의 순수한 가우시안 노이즈에 가까워져 의미론적 신호가 존재하지 않으며, 이로 인해 증류 목표가 무의미해진다. 이를 해결하기 위해 **타임스텝 게이트**를 도입한다.

$$w(t) = \mathbb{1}\left[t_{\min} \leq t \leq t_{\max}\right]$$

기본값으로 $t_{\min} = 0.1T$, $t_{\max} = 0.6T$로 설정한다. 이는 선행 연구에서 diffusion 모델 내부의 의미론적 특징이 가장 활성화되는 것으로 알려진 낮은-중간 노이즈 구간으로 증류 신호를 제한한다.

선택적으로, 하드 임계값 대신 부드러운 시그모이드 게이트를 사용할 수 있다.

$$w(t) = \sigma\left(-\frac{t - t_{\text{mid}}}{\beta}\right)$$

여기서 $t_{\text{mid}} = 0.4T$이며, $\beta$는 게이트의 날카로움을 제어한다.

#### 3.3.5 Total Training Objective

SDD의 전체 목적 함수는 다음과 같다.

$$\mathcal{L}_{\text{SDD}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot w(t) \cdot \mathcal{L}_{\text{DINO}}(p_t, p_s)$$

여기서

$$\mathcal{L}_{\text{MSE}} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

$$\mathcal{L}_{\text{DINO}} = -\sum_{k=1}^K p_t^{(k)} \log p_s^{(k)}$$

이며, $\gamma = 0.5$는 두 목적 함수의 균형을 조절하는 스칼라 가중치이다. 경사는 학생 네트워크를 통해서만 역전파되며, 교사는 EMA를 통해서만 갱신된다.

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
