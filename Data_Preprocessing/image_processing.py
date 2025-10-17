#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === GStreamer가 경로를 파이프라인으로 오인하지 않게 끄기 (cv2 import 전에) ===
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

import cv2
from pathlib import Path

# --------------------
# 설정
# --------------------
# 입력 파일은 project/DATA_PROCESSING/Edit/<FILE_NAME>.mp4 에 있다고 하셨습니다.
FILE_NAME   = "UJ_VE101603"            # 확장자 제외 (예: "광우_VE101601" 또는 "test")
TARGET_FPS  = 5.0               # 출력 FPS
EXT         = ".mp4"            # 입력 확장자

# 프로젝트 루트 (이 파일 기준 .../project)
ROOT        = Path(__file__).resolve().parents[1]
IN_DIR      = ROOT / "DATA_PROCESSING" / "Edit"
OUT_V_DIR   = ROOT / "DATA_PROCESSING" / "output_video"
OUT_F_DIR   = ROOT / "DATA_PROCESSING" / "output_frame" / FILE_NAME

INPUT_VIDEO  = (IN_DIR / f"{FILE_NAME}{EXT}").resolve()
OUTPUT_VIDEO = (OUT_V_DIR / f"{FILE_NAME}_5fps.mp4").resolve()

# --------------------
# 헬퍼: 백엔드 순차 재시도
# --------------------
def open_video_with_fallback(path: Path) -> cv2.VideoCapture:
    """
    시도 순서: ANY → FFMPEG → MSMF → DSHOW
    pip의 opencv-python 휠을 쓰면 보통 ANY에서 FFMPEG로 열립니다.
    """
    candidates = [
        ("ANY",    cv2.CAP_ANY),
        ("FFMPEG", cv2.CAP_FFMPEG),
        ("MSMF",   cv2.CAP_MSMF),
        ("DSHOW",  cv2.CAP_DSHOW),
    ]
    tried = []
    for name, backend in candidates:
        cap = cv2.VideoCapture(str(path), backend)
        ok = cap.isOpened()
        tried.append(f"{name}:{'OK' if ok else 'FAIL'}")
        if ok:
            print(f"[INFO] opened with backend: {name}")
            return cap
    raise RuntimeError(f"비디오를 열 수 없습니다 (tried {', '.join(tried)}): {path}")

# --------------------
# 준비
# --------------------
print("[DBG] cwd:", os.getcwd())
print("[DBG] INPUT:", INPUT_VIDEO)
print("[DBG] INPUT exists:", INPUT_VIDEO.exists())

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_VIDEO}")

OUT_V_DIR.mkdir(parents=True, exist_ok=True)
OUT_F_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# 입력 열기
# --------------------
cap = open_video_with_fallback(INPUT_VIDEO)

# 입력 정보
orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (w, h)
print(f"[INFO] input fps≈{orig_fps:.3f}, size={size}")

# --------------------
# 출력 비디오 준비 (5fps)
# --------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, TARGET_FPS, size)
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"출력 비디오를 생성할 수 없습니다: {OUTPUT_VIDEO}")

emit_period = 1.0 / TARGET_FPS   # 0.2s 간격
next_emit_t = 0.0
saved_count = 0
last_frame  = None
last_t_sec  = 0.0
sampled_frames = []

# --------------------
# 타임스탬프 기반 샘플링
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 타임스탬프(초)
    t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if t_sec <= 0:
        # 일부 코덱에서 초반 0 보정: 프레임 인덱스로 근사
        idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if idx >= 0 and orig_fps > 0:
            t_sec = float(idx) / float(orig_fps)

    last_t_sec = t_sec
    last_frame = frame

    # 목표 시각(next_emit_t)을 초과/충족할 때마다 프레임 선택
    while t_sec + 1e-6 >= next_emit_t:
        sampled_frames.append(frame.copy())
        saved_count += 1
        next_emit_t += emit_period

cap.release()

# --------------------
# 길이 보정: round(원본길이 * TARGET_FPS) 프레임 개수로 맞추기
# --------------------
target_frame_count = max(1, int(round(last_t_sec * TARGET_FPS)))

if saved_count < target_frame_count and last_frame is not None:
    # 부족분은 마지막 프레임으로 패딩
    sampled_frames.extend([last_frame.copy() for _ in range(target_frame_count - saved_count)])
    saved_count = target_frame_count
elif saved_count > target_frame_count:
    # 과한 경우는 잘라냄(가변 FPS에서 드물게 발생)
    sampled_frames = sampled_frames[:target_frame_count]
    saved_count = target_frame_count

# --------------------
# 비디오/이미지 저장
# --------------------
for i, f in enumerate(sampled_frames):
    writer.write(f)
    (OUT_F_DIR / f"frame_{i:05d}.jpg").write_bytes(cv2.imencode(".jpg", f)[1].tobytes())

writer.release()

print(f"[DONE] 원본 길이(추정): {last_t_sec:.3f}s")
print(f"[DONE] 출력 프레임 수: {saved_count} @ {TARGET_FPS}fps → 길이≈{saved_count/TARGET_FPS:.3f}s")
print(f"[DONE] 출력 비디오: {OUTPUT_VIDEO}")
print(f"[DONE] 프레임 폴더:  {OUT_F_DIR}")
