# 🪨 Rock Type Classification with Multi-GPU DDP Training

 **Dacon 자갈 암석 분류 대회**를 위해 진행된 분류 모델 학습 파이프라인
 
Jetson AGX Orin 기반 멀티 노드 분산 학습(DDP) 환경 - 최종 점수 0.84178

## 🚀 주요 특징

- **모델 아키텍처**
  - `timm` 라이브러리 기반의 `resnet101` 백본
  - `conv1 ~ layer2`는 Freeze, `layer3~4`, `fc`만 fine-tuning

- **데이터 전처리**
  - `PadSquare` 클래스를 활용해 정사각형 padding 적용
  - `Albumentations` 기반 다양한 augmentation
    - Horizontal/Vertical Flip, ShiftScaleRotate, Blur, CoarseDropout 등

- **학습 기법**
  - `DistributedDataParallel`을 활용한 **멀티 노드 DDP 학습**
  - `CrossEntropyLoss`에 `class_weight`와 `label_smoothing` 적용
  - `GradScaler` + `autocast()` 기반 AMP (mixed precision training)
  - `ReduceLROnPlateau` 학습률 스케줄러 사용
  - `LivePlot`을 활용한 실시간 학습 그래프 시각화

- **검증 및 평가**
  - `macro F1-score` 기반 성능 평가
  - 클래스 비율을 고려한 Stratified Split

- **TTA (Test Time Augmentation)**
  - 테스트 시 좌우반전 포함한 TTA 적용
  - 최종 결과를 `.csv` 형식으로 저장
  - 이후 점수 문제로 삭제

## 🧩 학습 파이프라인

```text
[Dataset] 
  ↓
[Albumentations Augmentation]
  ↓
[CustomDataset]
  ↓
[DistributedDataLoader]
  ↓
[ResNet101 모델 (conv1~layer2 freeze)]
  ↓
[AMP + DDP 학습]
  ↓
[Validation: macro F1] 
  ↓
[최적 모델 저장 및 TTA Inference]

```


## rock_classification_baseline.py
- **score**: 0.67288  
- 기존 baseline code 학습, Jetson AGX Orin 1대  
- mobilenetv3 훈련  
- [개발환경 구축 및 baseline code](https://cafe.daum.net/SmartRobot/RoVa/2202)

---

## rock_classification_multi_gpu_v1.py
- **score**: 0.73568  
- Jetson 기반 Multi-Node 학습, Jetson AGX Orin 4대  
- resnet101 훈련  
- [Jetson 기반 Multi-Node 학습](https://cafe.daum.net/SmartRobot/RoVa/2206)

---

## rock_classification_multi_gpu_v2.py
- **score**: 0.76311  
- class weight, augmentation 추가  
- resnet50 훈련  
- [class weight, augmentation](https://cafe.daum.net/SmartRobot/RoVa/2216)

---

## rock_classification_multi_gpu_v3.py
- **score**: 0.78917  
- Two-Stage Fine-Tuning 적용, TTA 적용  
- resnet50 훈련  
- [Two-Stage Fine-Tuning](https://cafe.daum.net/SmartRobot/RoVa/2222)

---

## rock_classification_multi_gpu_v4.py
- **score**: 0.8195  
- Two-Stage Fine-Tuning 적용  
- resnet101 훈련  
- [Two-Stage Fine-Tuning resnet101](https://cafe.daum.net/SmartRobot/RoVa/2227)

---

## rock_classification_multi_gpu_v5.py
- **score**: 0.84178  
- TTA 삭제, Window RTX4070 super 환경에서 학습, Two-Stage Fine-Tuning 유지  
- resnet101 훈련  

