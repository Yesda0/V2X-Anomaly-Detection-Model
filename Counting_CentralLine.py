import argparse
import glob
import os
import sys
from typing import List, Tuple, Set

import cv2
import numpy as np

# ultralytics YOLOv8
try:
    from ultralytics import YOLO
except Exception:
    print("Ultralytics가 설치되어 있지 않습니다. `pip install ultralytics` 후 다시 시도하세요.")
    raise

# -----------------------------
# 설정값(필요하면 바꿔쓰기)
# -----------------------------
CENTERLINE_HINTS: Tuple[str, ...] = ("central_line")
VEHICLE_HINTS: Tuple[str, ...]    = ("car")

# (2) 점수/IoU 임계값
DEFAULT_CONF = 0.6
DEFAULT_IOU  = 0.6

# -----------------------------
# 유틸
# -----------------------------
def find_class_ids_by_hints(names: dict, hints: Tuple[str, ...]) -> Set[int]:
    """클래스 이름 딕셔너리에서 hints(부분 문자열)가 포함된 클래스 id들을 찾아 반환."""
    found = set()
    for cid, cname in names.items():
        low = str(cname).lower()
        if any(h in low for h in hints):
            found.add(cid)
    return found

def robust_centerline_lookup(mask: np.ndarray, min_band_width: int = 3) -> dict:
    """
    바이너리 마스크에서 y->x_mid lookup을 만든다.
    - 각 y행에서 mask>0인 연속 구간들 중 '가장 넓은 구간'을 선택하고,
      그 좌우 경계의 중점 (left+right)/2 를 중앙선 x로 사용.
    - 띠 폭이 너무 좁으면(노이즈) 제외.
    - 사소한 구멍/돌기는 MORPH_CLOSE로 1차 정리.
    """
    H, W = mask.shape[:2]
    lookup = {}

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    for y in range(H):
        row = mask[y] > 0
        if not row.any():
            continue
        xs = np.where(row)[0]
        # 연속 구간 분할
        gaps = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, gaps + 1]
        ends   = np.r_[gaps, xs.size - 1]

        best_w = -1
        best_pair = None
        for s, e in zip(starts, ends):
            left, right = xs[s], xs[e]
            width = right - left + 1
            if width > best_w:
                best_w = width
                best_pair = (left, right)

        if best_pair is None or best_w < min_band_width:
            continue

        left, right = best_pair
        lookup[y] = float(0.5 * (left + right))

    return lookup

def smooth_centerline_lookup(lookup: dict, med_ks: int = 21, ema_alpha: float = 0.15) -> dict:
    """
    lookup(y->x)을 y순으로 정렬해 1D 스무딩.
    - median filter(홀수 window)
    - EMA(지수 이동 평균)
    """
    if not lookup:
        return lookup
    ys = np.array(sorted(lookup.keys()))
    xs = np.array([lookup[y] for y in ys], dtype=float)

    # Median filter
    if med_ks > 1 and med_ks % 2 == 1 and xs.size >= med_ks:
        xs_pad = np.pad(xs, (med_ks//2, med_ks//2), mode='edge')
        xs_med = np.array([np.median(xs_pad[i:i+med_ks]) for i in range(xs.size)], dtype=float)
    else:
        xs_med = xs

    # EMA
    xs_smooth = np.empty_like(xs_med)
    xs_smooth[0] = xs_med[0]
    for i in range(1, xs_med.size):
        xs_smooth[i] = ema_alpha * xs_med[i] + (1 - ema_alpha) * xs_smooth[i-1]

    return {int(y): float(x) for y, x in zip(ys, xs_smooth)}

def clamp_dx(lookup: dict, max_dx: int = 8) -> dict:
    """연속 y 간 x 변화량을 제한하여 지그재그를 추가로 억제."""
    if not lookup:
        return lookup
    ys = sorted(lookup.keys())
    xs = [lookup[y] for y in ys]
    for i in range(1, len(xs)):
        delta = xs[i] - xs[i-1]
        if delta >  max_dx: xs[i] = xs[i-1] + max_dx
        if delta < -max_dx: xs[i] = xs[i-1] - max_dx
    return {y: float(x) for y, x in zip(ys, xs)}

def merge_lookups(lookups: List[dict]) -> dict:
    """여러 중앙선 lookup을 평균으로 병합."""
    merged, counter = {}, {}
    for lu in lookups:
        for y, x in lu.items():
            merged[y]  = merged.get(y, 0.0) + x
            counter[y] = counter.get(y, 0) + 1
    for y in list(merged.keys()):
        merged[y] /= counter[y]
    return merged

def x_on_centerline(lookup: dict, y: int, H: int, fallback_x: int, search_radius: int = 40) -> float:
    """해당 y에서 중앙선 x를 반환. 없으면 근처 y 탐색, 그마저 없으면 fallback_x."""
    if y in lookup:
        return lookup[y]
    for dy in range(1, search_radius + 1, 2):
        yu = min(H - 1, y + dy)
        yd = max(0, y - dy)
        if yu in lookup: return lookup[yu]
        if yd in lookup: return lookup[yd]
    return float(fallback_x)

def draw_poly_from_lookup(img: np.ndarray, lookup: dict, step: int = 6, color=(0, 165, 255), thickness: int = 3):
    """lookup(y->x)으로부터 폴리라인 근사 시각화."""
    if not lookup:
        return
    ys = sorted(lookup.keys())
    pts = []
    for y in ys[::step]:
        x = int(round(lookup[y]))
        pts.append((x, y))
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

# -----------------------------
# 메인 로직
# -----------------------------
def process_image(model: YOLO, img_path: str, conf: float, iou: float, imgsz: int,
                  center_ids: Set[int], vehicle_ids: Set[int], out_dir: str) -> Tuple[int, int]:
    """한 장 처리: 중앙선 기준 좌/우 차량 수 반환하고 결과 이미지 저장."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 이미지를 읽을 수 없습니다: {img_path}")
        return 0, 0
    H, W = img.shape[:2]
    fallback_x = W // 2

    # 추론
    res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    # 중앙선 마스크들 수집 → 견고한 lookup 생성
    center_lookups = []
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()  # (N, h, w)
        clses = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.array([])
        for i in range(masks.shape[0]):
            cid = int(clses[i]) if i < clses.shape[0] else -1
            if cid in center_ids:
                # 원본 크기로 resize & binarize
                m = (masks[i] > 0.5).astype(np.uint8) * 255
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                # 끊김 보강: CLOSE + DILATE
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)), iterations=1)

                lu = robust_centerline_lookup(m, min_band_width=3)
                if lu:
                    center_lookups.append(lu)

    # 여러 중앙선이 있으면 평균 병합 → 스무딩 → dx 제한
    center_lookup_raw = merge_lookups(center_lookups) if center_lookups else {}
    center_lookup = smooth_centerline_lookup(center_lookup_raw, med_ks=21, ema_alpha=0.15)
    center_lookup = clamp_dx(center_lookup, max_dx=8)

    # 차량 박스들 수집 및 좌/우 카운트
    left, right = 0, 0
    if res.boxes is not None:
        bxyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        bcls  = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cid in zip(bxyxy, bcls):
            if cid not in vehicle_ids:
                continue
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            x_mid = x_on_centerline(center_lookup, cy, H, fallback_x)
            if cx < x_mid:
                left += 1
                color = (255, 120, 120)
            else:
                right += 1
                color = (120, 220, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (cx, cy), 3, color, -1)

    # 중앙선 시각화
    draw_poly_from_lookup(img, center_lookup, step=4, color=(0, 165, 255), thickness=4)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + "_counted.jpg")
    cv2.imwrite(out_path, img)
    print(f"[OK] {os.path.basename(img_path)}  →  LEFT {left}  |  RIGHT {right}  (저장: {out_path})")
    return left, right

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  type=str, default="yolov8n-seg.pt", help="세그 모델 가중치(pt)")
    parser.add_argument("--img_glob", type=str, default="*.jpg", help='처리할 이미지 글롭 (쉼표로 여러개: \"*.jpg,*.png\")')
    parser.add_argument("--conf",     type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou",      type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz",    type=int,   default=1024, help="추론 해상도(훈련과 동일 권장)")
    parser.add_argument("--out_dir",  type=str,   default="counted_out")
    args = parser.parse_args()

    # 모델 로드
    model = YOLO(args.weights)
    names = model.names  # dict: id -> name

    # 중앙선/차량 클래스 자동 식별
    center_ids  = find_class_ids_by_hints(names, CENTERLINE_HINTS)
    vehicle_ids = find_class_ids_by_hints(names, VEHICLE_HINTS)

    # 자동 식별 실패 시 안내
    if not center_ids:
        print("[WARN] 중앙선 클래스를 자동으로 찾지 못했습니다. 모델 클래스:", names)
        print("       CENTERLINE_HINTS를 조정하거나 중앙선 class id를 수동 지정하세요. 예) center_ids = {0}")
    if not vehicle_ids:
        print("[WARN] 차량 클래스를 자동으로 찾지 못했습니다. 모델 클래스:", names)
        print("       VEHICLE_HINTS를 조정하거나 차량 class id를 수동 지정하세요. 예) vehicle_ids = {1,2}")

    # 이미지 목록
    patterns = [p.strip() for p in args.img_glob.split(",")]
    img_list: List[str] = []
    for p in patterns:
        img_list.extend(glob.glob(p))
    img_list = sorted(set(img_list))
    if not img_list:
        print(f"[INFO] 이미지가 없습니다. 패턴: {args.img_glob}")
        sys.exit(0)

    # 처리
    total_left = total_right = 0
    for path in img_list:
        l, r = process_image(model, path, args.conf, args.iou, args.imgsz, center_ids, vehicle_ids, args.out_dir)
        total_left  += l
        total_right += r

    print(f"\n=== 전체 합계 ===  LEFT: {total_left}  |  RIGHT: {total_right}")

if __name__ == "__main__":
    main()
