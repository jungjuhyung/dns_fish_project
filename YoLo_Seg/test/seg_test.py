#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from ultralytics import YOLO

# 경로 설정
MODEL_VERSION = f"1017FirstTrain"
TEST_DATA = "TestData_Half"
TEST_DATA_YAML  = f"./datasets/Segmentation_Data/{TEST_DATA}/data.yaml"
TEST_DATA_SOURCE  = f"./datasets/Segmentation_Data/{TEST_DATA}/test/images"

MODEL_PATH = f"./YoLo_Seg/train_results/{MODEL_VERSION}/weights/best.pt"   # 학습된 모델 경로
SAVE_DIR   = f"./YoLo_Seg/test_results/{MODEL_VERSION}_Res"                                 # 혼동행렬 저장 폴더

# 모델 로드
model = YOLO(MODEL_PATH)

# 테스트셋 평가 (YOLO 자체 confusion matrix 사용)
metrics = model.val(
    data=TEST_DATA_YAML,
    split='test',        # 또는 'test' 데이터셋이 있다면 'test'
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=0,          # GPU 또는 'cpu'
    save_json=False,
    verbose=True,
    project=SAVE_DIR,       # 혼동행렬 등 결과를 이 경로에 저장
    save=True
)

model.predict(source=TEST_DATA_SOURCE, save=True, project=SAVE_DIR, name="Test_Pred")