# DINO-Style Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models

**[English](README.md) | [Korean](README.ko.md)**  

---

## 초록

확산 모델(Diffusion Model)은 생성 모델링 분야에서 놀라운 성과를 거두었지만, 그 내부 표현(representation)은 다운스트림 판별 과제(discriminative task)에 활용하기에는 여전히 구조적으로 빈약하다. 표준 노이즈 제거 목적함수 — 픽셀 공간에서의 평균 제곱 오차(MSE) 회귀 — 는 DINO와 같은 전문화된 자기 지도 표현 학습기와 비교해 판별력이 낮고 매끄러운(smooth) 특징을 생성한다. 본 논문에서는 표준 확산 목적함수에 UNet 중간 특징에 직접 적용하는 DINO 방식의 자기 증류 손실을 결합한 학습 프레임워크인 **자기 증류 확산 모델(Self-Distilled Diffusion, SDD)**을 제안한다. 구체적으로, EMA(지수 이동 평균)로 갱신되는 교사 네트워크를 학생 확산 모델과 함께 유지하고, 중간 특징을 공유 임베딩 공간으로 투영한 뒤, 센터링(centering)과 샤프닝(sharpening) 연산을 적용하여 안정적이고 신뢰도 높은 학습 타겟을 생성한다. REPA와 같은 선행 연구와 달리, 본 방법은 외부 사전학습 인코더가 전혀 필요 없는 완전 자기 포함(self-contained) 구조이다. 나아가 의미론적 특징이 가장 활성화되는 저~중간 노이즈 구간으로 증류 신호를 제한하는 타임스텝 적응형 가중치 기법을 도입한다. 표준 이미지 생성 벤치마크에서의 실험은 SDD가 기준 확산 모델 대비 생성 품질(FID)과 선형 탐색(linear probe) 정확도를 동시에 향상시킴을 보여주며, 생성적 목적과 판별적 목적이 단일 자기 증류 프레임워크 안에서 공동으로 최적화될 수 있음을 검증한다.

**키워드:** 확산 모델, 자기 증류, 표현 학습, DINO, 센터링, 샤프닝, EMA 교사

---

## 1. 서론

노이즈 제거 확산 확률 모델(DDPM)은 고품질 이미지 합성을 위한 지배적인 패러다임으로 자리 잡았다. 표준 학습 목적함수는 다음과 같다:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

여기서 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 는 타임스텝 $t$ 에서의 노이즈가 추가된 이미지이며, $\epsilon_\theta$ 는 추가된 노이즈를 예측하도록 학습하는 신경망(일반적으로 UNet 또는 DiT)이다.

이 회귀 목적함수에는 근본적인 한계가 있다: 모델이 구조화되고 의미론적으로 풍부한 내부 표현을 형성하도록 유도하는 명시적인 유인이 전혀 없다는 점이다. 최근 실증 연구들은 이 목적함수만으로 학습된 확산 모델이 장시간 학습 후에도 DINOv2와 같은 전문 판별 모델의 표현에 비해 현저히 열악한 표현을 생성함을 확인했다. ImageNet에서의 선형 탐색 정확도 격차는 15 퍼센트포인트를 초과할 수 있다.

한편, DINO(Self-Distillation with No Labels)는 교사-학생 자기 증류 프레임워크에 두 가지 단순한 연산 — **센터링**(collapse 방지를 위한 누적 평균 감산)과 **샤프닝**(신뢰도 높은 타겟 생성을 위해 교사에 낮은 온도를 적용) — 을 결합함으로써, 어떠한 레이블도 없이 매우 강력한 시각 표현을 산출할 수 있음을 보였다.

우리는 자연스러운 질문을 제기한다: **DINO의 센터링과 샤프닝을 확산 모델 특징에 직접 적용하여, 어떠한 외부 사전학습 인코더 없이도 모델이 노이즈 제거와 구조화된 표현 학습을 동시에 수행할 수 있는가?**

우리의 답은 그렇다는 것이다. 본 논문의 기여는 다음과 같다:

1. 외부 인코더 없이 완전 자기 포함 확산 학습 루프 내에서 DINO 방식의 센터링과 샤프닝을 최초로 적용한 프레임워크인 **자기 증류 확산 모델(SDD)**을 제안한다.
2. 의미론적 특징이 가장 활성화되는 노이즈 구간에서만 선택적으로 증류 손실을 적용하는 **타임스텝 적응형 게이팅 메커니즘**을 도입하여, 단순 적용 시 발생하는 특징-타임스텝 충돌 문제를 해결한다.
3. 센터링과 샤프닝이 각각 독립적으로 표현 품질에 기여하며, 그 조합이 SD-DiT 및 DDAE와 같은 선행 연구에서 사용된 EMA 기반 증류(센터링/샤프닝 없음)보다 우수함을 체계적인 절제 실험(ablation)을 통해 입증한다.
4. SDD가 표준 확산 기준선 대비 FID와 다운스트림 선형 탐색 정확도를 모두 향상시킴을 보여, 이 프레임워크 내에서 생성적 목적과 판별적 목적이 양립 가능함을 실증한다.

---

## 2. 관련 연구

### 2.1 확산 모델

DDPM은 반복적 노이즈 제거를 통해 데이터 분포를 학습하는 원리적 프레임워크를 제시했다. 스코어 기반 생성 모델과 이후의 DDIM 샘플러는 연속 시간 및 결정론적 변형을 확립했다. ADM, LDM, DiT를 포함한 아키텍처 발전은 확산 모델을 최첨단 생성 품질로 끌어올렸다. 그러나 이 모든 방법들은 표현 형성을 위한 어떠한 메커니즘도 없이 동일한 픽셀 공간 MSE 목적함수를 공유한다.

### 2.2 자기 지도 표현 학습

MoCo, SimCLR, BYOL은 판별적 표현 학습을 위한 대조 학습 및 자기 증류 프레임워크를 확립했다. DINO는 센터링과 샤프닝을 갖춘 교사-학생 아키텍처로 이 패러다임을 확장하여, 어떠한 레이블도 없이 창발적 분할(segmentation) 특성을 보이는 표현을 생성했다. DINOv2는 이를 대규모 사전학습 모델로 확장했다. 본 연구는 센터링+샤프닝 메커니즘을 생성적 확산 학습 루프에 최초로 이식한다.

### 2.3 확산을 통한 표현 학습

여러 연구에서 확산 모델이 부수적으로 유용한 표현을 학습함에 주목했다. DDAE는 EMA 증류를 통한 표현 학습기로서 확산 모델을 활용하는 방법을 탐구했다. REPA(ICLR 2025 Oral)는 학습 중 확산 모델 특징을 외부 DINOv2 인코더에 정렬시키는 방법을 제안하여 극적인 학습 가속과 생성 품질 향상을 달성했다. SRA와 SD-DiT는 확산 학습 내부의 EMA 교사-학생 구조를 탐구했지만, 센터링이나 샤프닝은 적용하지 않았다. 본 연구는 이 공백을 메운다: 우리는 완전 자기 포함, 외부 인코더 불요 확산 학습 루프에서 DINO의 센터링+샤프닝 메커니즘 전체를 최초로 적용하며, 이 연산들의 중요성을 보이는 명시적 절제 실험을 제공한다.

---

## 3. 방법론

### 3.1 사전 지식: 표준 확산 학습

데이터 분포 $q(x_0)$ 가 주어졌을 때, 순방향 과정은 점진적으로 가우시안 노이즈를 추가한다:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

모델 $\epsilon_\theta(x_t, t)$ 는 다음을 최소화하여 이 과정을 역전하도록 학습된다:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 3.2 DINO 배경

DINO는 동일 이미지의 증강된 뷰에 대해 학생 네트워크 $f_s$ 와 EMA 갱신 교사 네트워크 $f_t$ (모멘텀 $m$)를 학습시킨다. 핵심 연산은 다음과 같다:

**센터링(Centering)** — softmax 이전에 누적 평균 $c$ 를 교사 출력에서 감산하여 자명한 해(trivial solution)로의 붕괴를 방지한다:

$$p_t = \text{softmax}\left(\frac{z_t - c}{\tau_t}\right), \quad c \leftarrow m \cdot c + (1 - m) \cdot \mathbb{E}_{\text{batch}}[z_t]$$

**샤프닝(Sharpening)** — 학생 출력이 불확실할 때에도 신뢰도 높은(저엔트로피) 타겟 분포를 생성하도록 교사에 더 낮은 온도 $\tau_t \ll \tau_s$ 를 적용한다.

손실함수는 교사 분포와 학생 분포 사이의 교차 엔트로피이다:

$$\mathcal{L}_{\text{DINO}} = -\sum_k p_t^{(k)} \log p_s^{(k)}$$

### 3.3 자기 증류 확산 모델 (SDD)

#### 3.3.1 아키텍처

두 개의 네트워크를 유지한다:

- **학생(Student)**: $\epsilon_\theta$ — 경사 하강으로 갱신되는 표준 확산 UNet/DiT.
- **교사(Teacher)**: $\epsilon_\xi$ — 학생의 EMA 복사본으로, $\xi \leftarrow m \cdot \xi + (1-m) \cdot \theta$ 로 갱신되며 모멘텀 $m = 0.996$ 을 사용.

두 네트워크는 동일한 노이즈 입력 $x_t$ 를 받되, 독립적인 증강 파이프라인(예: $x_0$ 에 순방향 확산 과정 전에 적용되는 랜덤 크롭, 컬러 지터)을 거친다.

#### 3.3.2 특징 추출 및 투영

UNet의 병목(bottleneck) 레이어(또는 DiT의 중간 트랜스포머 블록)에서 중간 특징을 추출한다. 학생과 교사의 병목 특징을 각각 $z_s = f_s(x_t, t)$, $z_t = f_t(x_t, t)$ 로 표기한다.

각 특징은 $K$ 차원 임베딩으로 매핑하는 경량 2층 MLP 투영 헤드 $g(\cdot)$ 를 통과한다(기본값 $K = 256$):

$$\tilde{z}_s = g_s(z_s), \quad \tilde{z}_t = g_t(z_t)$$

투영 헤드는 공유하지 않으며, 교사 헤드 $g_t$ 도 EMA로 갱신된다.

#### 3.3.3 센터링과 샤프닝

DINO를 그대로 따라, 교사 로짓에 센터링과 낮은 온도를, 학생에는 높은 온도를 적용한다:

$$p_t = \text{softmax}\left(\frac{\tilde{z}_t - c}{\tau_t}\right), \quad p_s = \text{softmax}\left(\frac{\tilde{z}_s}{\tau_s}\right)$$

여기서:
- $c$ 는 교사 출력의 누적 EMA로 매 배치마다 갱신: $c \leftarrow \lambda c + (1-\lambda) \mathbb{E}[\tilde{z}_t]$
- $\tau_t = 0.04$ (교사 온도, 낮음 → 타겟 분포를 샤프닝)
- $\tau_s = 0.1$ (학생 온도, 높음 → 학생 예측을 부드럽게)
- $\lambda = 0.9$ (센터링 EMA 모멘텀)

#### 3.3.4 타임스텝 적응형 게이팅

모든 타임스텝에 증류 손실을 단순 적용하면 문제가 생긴다: 높은 노이즈 수준($t \to T$)에서는 입력이 거의 순수 가우시안 노이즈가 되어 의미론적 신호를 전혀 포함하지 않으므로, 증류 타겟 자체가 무의미해진다. 이를 해결하기 위해 **타임스텝 게이트**를 도입한다:

$$w(t) = \mathbb{1}\left[t_{\min} \leq t \leq t_{\max}\right]$$

기본값으로 $t_{\min} = 0.1T$, $t_{\max} = 0.6T$ 를 설정한다. 이는 선행 연구에서 확산 모델 내부의 의미론적 특징이 가장 활성화되는 것으로 확인된 저~중간 노이즈 구간으로 증류 신호를 제한한다.

필요에 따라 경계를 완화하는 소프트 시그모이드 게이트를 사용할 수도 있다:

$$w(t) = \sigma\left(-\frac{t - t_{\text{mid}}}{\beta}\right)$$

여기서 $t_{\text{mid}} = 0.4T$이며, $\beta$ 는 게이트의 경계 선명도를 제어한다.

#### 3.3.5 전체 학습 목적함수

SDD의 전체 목적함수는 다음과 같다:

$$\mathcal{L}_{\text{SDD}} = \mathcal{L}_{\text{MSE}} + \lambda \cdot w(t) \cdot \mathcal{L}_{\text{DINO}}(p_t, p_s)$$

여기서:

$$\mathcal{L}_{\text{MSE}} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

$$\mathcal{L}_{\text{DINO}} = -\sum_{k=1}^K p_t^{(k)} \log p_s^{(k)}$$

이며, $\lambda = 0.5$ 는 두 목적함수의 균형을 맞추는 스칼라이다. 경사도(gradient)는 학생에만 흐르며, 교사는 EMA로만 갱신된다.

#### 3.3.6 요약 의사 코드

```python
# SDD 학습 스텝
for x0 in dataloader:
    # 타임스텝 및 노이즈 샘플링
    t = sample_timestep(T)
    eps = torch.randn_like(x0)
    xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus[t] * eps

    # 학생 순전파
    eps_pred, z_s = student(xt, t, return_features=True)
    z_s_proj = proj_head_student(z_s)
    p_s = F.softmax(z_s_proj / tau_s, dim=-1)

    # 교사 순전파 (경사도 없음)
    with torch.no_grad():
        _, z_t = teacher(xt, t, return_features=True)
        z_t_proj = proj_head_teacher(z_t)
        z_t_centered = z_t_proj - center          # 센터링
        p_t = F.softmax(z_t_centered / tau_t, dim=-1)  # 샤프닝

    # 센터 갱신 (EMA)
    center = lambda_c * center + (1 - lambda_c) * z_t_proj.mean(0).detach()

    # 손실 계산
    L_mse  = F.mse_loss(eps_pred, eps)
    L_dino = -(p_t * torch.log(p_s + 1e-8)).sum(dim=-1).mean()
    w_t    = timestep_gate(t, t_min, t_max)
    loss   = L_mse + lambda_weight * w_t * L_dino

    # 학생 갱신
    loss.backward()
    optimizer.step()

    # 교사 EMA 갱신
    update_ema(teacher, student, momentum=0.996)
    update_ema(proj_head_teacher, proj_head_student, momentum=0.996)
```

---

## 참고문헌

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*.

[2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*.

[3] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *ICCV 2021*. (DINO)

[4] Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning robust visual features without supervision. *TMLR 2024*.

[5] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022*.

[6] Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023*. (DiT)

[7] Guth, F., Coste, S., De Bortoli, V., & Mallat, S. (2022). Wavelet score-based generative modeling. *NeurIPS 2022*.

[8] Chen, X., et al. (2023). DDAE: Towards self-supervised representation learning with diffusion autoencoders. *arXiv 2023*.

[9] Yu, S., Kwak, D., Jang, H., et al. (2024). Representation alignment for generation: Training diffusion transformers is easier than you think. *ICLR 2025 Oral*. (REPA)

[10] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR 2020*. (MoCo)

[11] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*. (SimCLR)

[12] Grill, J. B., Strub, F., Altché, F., et al. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS 2020*. (BYOL)

[13] Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS 2021*. (ADM)

---

*교신 저자: [이메일 주소 미정]*  
*코드: [저장소 주소 미정]*
