#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
from pathlib import Path
from ultralytics import YOLO

# 실험 이름과 설정 파일 경로
train_name = "train_bb"
config_path = f"./YoLo_BB/config_bb/{train_name}.yaml"

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    # 1) 설정 로드
    cfg = load_config(config_path)

    # 2) 파라미터 언패킹
    model_path = cfg["model"]["pretrained"]   # 예: ./YoLo_Det/backbone/yolo11s.pt
    data_yaml  = cfg["data"]["yaml"]          # 예: ./YoLo_Det/config_det/train_data/data.yaml
    save_dir   = cfg["train"]["project"]      # 예: ./YoLo_Det/results/yolo11_det_results
    train_cfg  = cfg["train"]

    # 3) 모델 로드 (Detection 전용 가중치)
    model = YOLO(model_path)

    # 4) 학습
    model.train(
        task="detect",                 # ★ Detection
        data=data_yaml,                # YOLO bbox 데이터셋 YAML
        epochs=train_cfg.get("epochs", 100),
        imgsz=train_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch", 8),
        workers=train_cfg.get("workers", 0),
        device=train_cfg.get("device", 0),     # 0 또는 'cpu'
        optimizer=train_cfg.get("optimizer", "auto"),
        lr0=train_cfg.get("lr0", 0.01),
        lrf=train_cfg.get("lrf", 0.01),
        hsv_h=train_cfg.get("hsv_h", 0.015),
        hsv_s=train_cfg.get("hsv_s", 0.7),
        hsv_v=train_cfg.get("hsv_v", 0.4),
        degrees=train_cfg.get("degrees", 0.0),
        translate=train_cfg.get("translate", 0.1),
        scale=train_cfg.get("scale", 0.5),
        shear=train_cfg.get("shear", 0.0),
        perspective=train_cfg.get("perspective", 0.0),
        flipud=train_cfg.get("flipud", 0.0),
        fliplr=train_cfg.get("fliplr", 0.5),
        mosaic=train_cfg.get("mosaic", 0.0),
        patience=train_cfg.get("patience", 30),
        project=save_dir,
        name=train_cfg.get("name", "train"),               # runs/<project>/<name>
        exist_ok=True,
        verbose=True
    )

    # 5) 검증 (best.pt 기준)
    best = Path(save_dir) / train_cfg.get("name", "train") / "weights" / "best.pt"
    model = YOLO(str(best))
    metrics = model.val(
        task="detect",
        data=data_yaml,
        imgsz=train_cfg.get("imgsz", 640),
        device=train_cfg.get("device", 0)
    )
    print(metrics)

if __name__ == "__main__":
    main()
