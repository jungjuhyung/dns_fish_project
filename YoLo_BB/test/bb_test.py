#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from ultralytics import YOLO

# ✅ 경로 설정
MODEL_PATH = "./YoLo_BB/results/train_bb/weights/best.pt"   # 학습된 모델 경로
DATA_YAML  = "./YoLo_BB/config_bb/test_data/data.yaml"                       # 테스트셋이 포함된 data.yaml 경로
SAVE_DIR   = "./YoLo_BB/test_results/eval_confmat"                                 # 혼동행렬 저장 폴더

# ✅ 클래스 이름 불러오기
import yaml
with open(DATA_YAML, 'r', encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg["names"]
num_classes = len(class_names)

# ✅ 모델 로드
model = YOLO(MODEL_PATH)

# ✅ 테스트셋 평가 (YOLO 자체 confusion matrix 사용)
metrics = model.val(
    data=DATA_YAML,
    split='test',        # 또는 'test' 데이터셋이 있다면 'test'
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=0,          # GPU 또는 'cpu'
    save_json=False,
    verbose=True,
    project=SAVE_DIR,       # 👈 혼동행렬 등 결과를 이 경로에 저장
    save=True
)

# Ultralytics는 혼동행렬을 자동으로 results 디렉토리에 `confusion_matrix.png`로 저장함
print("==============")
print(metrics)  # precision, recall, mAP 등도 함께 출력됨
print(f"[INFO] Confusion matrix image is saved automatically in: {metrics.save_dir}")
print("==============")