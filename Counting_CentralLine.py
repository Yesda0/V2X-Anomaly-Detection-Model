import argparse
import glob
import os
import re
import sys
import csv
from typing import List, Tuple, Set, Dict, Optional

import cv2
import numpy as np

# -------- YOLOv8 (ultralytics) --------
try:
    from ultralytics import YOLO
except Exception:
    print("Ultralytics가 설치되어 있지 않습니다. `pip install ultralytics` 후 다시 시도하세요.")
    raise

# -------- PaddleOCR --------
try:
    from paddleocr import PaddleOCR
except Exception:
    print("PaddleOCR가 설치되어 있지 않습니다. `pip install paddlepaddle paddleocr` 후 다시 시도하세요.")
    raise

# =========================================================
# 설정값 (필요 시 사용자 환경에 맞게 수정)
# =========================================================

# (1) YOLOv8에서 중앙선/차량 클래스를 자동 식별하기 위한 힌트
CENTERLINE_HINTS: Tuple[str, ...] = ("central_line", "centerline", "lane", "중앙선")
VEHICLE_HINTS: Tuple[str, ...]    = ("car", "vehicle", "truck", "bus", "승용", "차량")

# (2) YOLOv8 추론 기본 파라미터
DEFAULT_CONF = 0.6
DEFAULT_IOU  = 0.6
DEFAULT_IMGSZ = 1024

# (3) YOLOv5(화살표/텍스트) 클래스 이름 (사용자 모델에 맞춰 수정)
#     Down_arrow / Up_arrow / text 로 학습되어 있다고 가정
V5_CLASS_NAMES = ['Down_arrow', 'Up_arrow', 'text']

# (4) 텍스트 허용어/동의어 사전 (필요시 확장)
ALLOWED_TEXTS = [
    '원효대교',
    '숙명여대',
    '공덕오거리',
    '삼각지',
]
CANONICAL_SYNONYMS = {
    '공덕오거리': ['공덕 오거리', '공덕오 거 리', '공덕 오 거 리'],
    '숙명여대': ['숙 명 여 대', '숙명 여자 대', '숙명 여 대'],
    '원효대교': ['원 효 대 교', '원효 대교'],
    '삼각지': ['삼 각 지'],
}

# =========================================================
# 유틸: 중앙선 처리 (YOLOv8 세그 마스크 → 중앙선 lookup)
# =========================================================
def find_class_ids_by_hints(names: Dict[int, str], hints: Tuple[str, ...]) -> Set[int]:
    found = set()
    for cid, cname in names.items():
        low = str(cname).lower()
        if any(h in low for h in hints):
            found.add(cid)
    return found

def robust_centerline_lookup(mask: np.ndarray, min_band_width: int = 3) -> Dict[int, float]:
    H, W = mask.shape[:2]
    lookup = {}
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    for y in range(H):
        row = mask[y] > 0
        if not row.any():
            continue
        xs = np.where(row)[0]
        gaps = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, gaps + 1]
        ends   = np.r_[gaps, xs.size - 1]
        best_w, best_pair = -1, None
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


def draw_poly_from_lookup(img: np.ndarray, lookup: dict, step: int = 6,
                          color=(0, 165, 255), thickness: int = 3):
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

def smooth_centerline_lookup(lookup: Dict[int, float], med_ks: int = 21, ema_alpha: float = 0.15) -> Dict[int, float]:
    if not lookup:
        return lookup
    ys = np.array(sorted(lookup.keys()))
    xs = np.array([lookup[y] for y in ys], dtype=float)
    if med_ks > 1 and med_ks % 2 == 1 and xs.size >= med_ks:
        xs_pad = np.pad(xs, (med_ks//2, med_ks//2), mode='edge')
        xs_med = np.array([np.median(xs_pad[i:i+med_ks]) for i in range(xs.size)], dtype=float)
    else:
        xs_med = xs
    xs_smooth = np.empty_like(xs_med)
    xs_smooth[0] = xs_med[0]
    for i in range(1, xs_med.size):
        xs_smooth[i] = ema_alpha * xs_med[i] + (1 - ema_alpha) * xs_smooth[i-1]
    return {int(y): float(x) for y, x in zip(ys, xs_smooth)}

def clamp_dx(lookup: Dict[int, float], max_dx: int = 8) -> Dict[int, float]:
    if not lookup:
        return lookup
    ys = sorted(lookup.keys())
    xs = [lookup[y] for y in ys]
    for i in range(1, len(xs)):
        delta = xs[i] - xs[i-1]
        if delta >  max_dx: xs[i] = xs[i-1] + max_dx
        if delta < -max_dx: xs[i] = xs[i-1] - max_dx
    return {y: float(x) for y, x in zip(ys, xs)}

def merge_lookups(lookups: List[Dict[int, float]]) -> Dict[int, float]:
    merged, counter = {}, {}
    for lu in lookups:
        for y, x in lu.items():
            merged[y]  = merged.get(y, 0.0) + x
            counter[y] = counter.get(y, 0) + 1
    for y in list(merged.keys()):
        merged[y] /= counter[y]
    return merged

def x_on_centerline(lookup: Dict[int, float], y: int, H: int, fallback_x: int, search_radius: int = 40) -> float:
    if y in lookup:
        return lookup[y]
    for dy in range(1, search_radius + 1, 2):
        yu = min(H - 1, y + dy)
        yd = max(0, y - dy)
        if yu in lookup: return lookup[yu]
        if yd in lookup: return lookup[yd]
    return float(fallback_x)

# =========================================================
# 유틸: YOLOv5 + OCR 로 화살표 ↔ 한글 매핑
# =========================================================
def normalize_text(text: str) -> str:
    if not text:
        return ''
    return re.sub(r"[^가-힣0-9]", "", text)

def expand_synonyms(canonical: str) -> list:
    variants = [canonical]
    variants.extend(CANONICAL_SYNONYMS.get(canonical, []))
    return list(dict.fromkeys(variants))

def choose_best_from_allowed(recognized: str, min_ratio: float = 0.65):
    import difflib
    rec_norm = normalize_text(recognized)
    if not rec_norm:
        return None, 0.0
    best_label, best_ratio = None, 0.0
    # 부분일치 우선
    for label in ALLOWED_TEXTS:
        for variant in expand_synonyms(label):
            v_norm = normalize_text(variant)
            if v_norm and (v_norm in rec_norm or rec_norm in v_norm):
                ratio = 0.999 if v_norm == rec_norm else 0.9
                if ratio > best_ratio:
                    best_label, best_ratio = label, ratio
    # 유사도 비교
    if best_label is None:
        for label in ALLOWED_TEXTS:
            for variant in expand_synonyms(label):
                v_norm = normalize_text(variant)
                ratio = difflib.SequenceMatcher(None, rec_norm, v_norm).ratio()
                if ratio > best_ratio:
                    best_label, best_ratio = label, ratio
    if best_ratio >= min_ratio:
        return best_label, best_ratio
    return None, best_ratio

def parse_ocr_result(result):
    texts = []
    try:
        if isinstance(result, list) and result and isinstance(result[0], dict):
            rec_texts = result[0].get('rec_texts') or []
            rec_scores = result[0].get('rec_scores') or []
            for t, s in zip(rec_texts, rec_scores):
                texts.append((str(t), float(s)))
            return texts
    except Exception:
        pass
    try:
        if isinstance(result, list) and result and isinstance(result[0], list):
            for item in result[0]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = str(item[1][0]); score = float(item[1][1])
                    texts.append((text, score))
    except Exception:
        pass
    return texts

def run_yolov5_detect(v5_dir: str, v5_weights: str, img_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    YOLOv5 detect.py를 서브프로세스로 실행해 labels를 파싱.
    반환: (arrows_data, texts_data)
      arrows_data: [{bbox, center, direction}]
      texts_data : [{bbox, center, raw, mapped, score, ratio}]
    """
    import subprocess, shutil

    # 이전 결과 삭제
    detect_dir = os.path.join(v5_dir, 'runs', 'detect')
    if os.path.exists(detect_dir):
        try:
            shutil.rmtree(detect_dir)
        except Exception:
            pass

    cmd = [
        sys.executable,
        os.path.join(v5_dir, "detect.py"),
        "--weights", v5_weights,
        "--source", img_path,
        "--conf", "0.25",
        "--save-txt"
    ]
    subprocess.run(cmd, check=False)

    result_folder = os.path.join(v5_dir, 'runs', 'detect', 'exp')
    label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    result_label_path = os.path.join(result_folder, 'labels', label_filename)

    img = cv2.imread(img_path)
    if img is None:
        return [], []
    H, W = img.shape[:2]

    def to_xyxy(parts_list):
        xc, yc, w, h = map(float, parts_list[1:5])
        x_min = max(0, int((xc - w / 2) * W))
        y_min = max(0, int((yc - h / 2) * H))
        x_max = min(W - 1, int((xc + w / 2) * W))
        y_max = min(H - 1, int((yc + h / 2) * H))
        return x_min, y_min, x_max, y_max

    def center_of(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    if not os.path.exists(result_label_path):
        return [], []

    ocr_model = PaddleOCR(lang='korean')
    arrows_data, texts_data = [], []
    with open(result_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue
            class_id = int(float(parts[0]))
            class_name = V5_CLASS_NAMES[class_id]
            bbox = to_xyxy(parts)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if 'arrow' in class_name:
                direction = "상행선" if class_name == 'Up_arrow' else "하행선"
                arrows_data.append({'bbox': bbox, 'center': center_of(bbox), 'direction': direction})
            elif class_name == 'text':
                x1, y1, x2, y2 = bbox
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                result = ocr_model.predict(roi)
                pairs = parse_ocr_result(result)
                if not pairs:
                    continue
                best_raw, best_score = max(pairs, key=lambda x: x[1])
                mapped, ratio = choose_best_from_allowed(best_raw)
                texts_data.append({
                    'bbox': bbox,
                    'center': center_of(bbox),
                    'raw': best_raw,
                    'mapped': mapped,
                    'score': float(best_score),
                    'ratio': float(ratio),
                })
    return arrows_data, texts_data

def match_direction_labels(arrows_data: List[Dict], texts_data: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    화살표 중심과 가까운 텍스트를 매칭 → 각 방향(하행/상행)의 대표 한글 라벨을 결정.
    반환: (down_korean, up_korean)
    """
    def euclidean(p, q):
        return ((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5

    down_labels, up_labels = [], []
    for a in arrows_data:
        a_center = a['center']
        # 가장 가까운 텍스트 하나 선택
        best_idx, best_d = None, 1e18
        for i, t in enumerate(texts_data):
            d = euclidean(a_center, t['center'])
            if d < best_d:
                best_d, best_idx = d, i
        if best_idx is None:
            continue
        t = texts_data[best_idx]
        label = t['mapped'] if t.get('mapped') else t.get('raw')
        if not label:
            continue
        if a['direction'] == '하행선':
            down_labels.append(label)
        else:
            up_labels.append(label)

    # 대표값 선택: 가장 많이 등장한 항목, 동률이면 첫 항목
    def pick_majority(labels: List[str]) -> Optional[str]:
        if not labels: 
            return None
        freq = {}
        for s in labels:
            freq[s] = freq.get(s, 0) + 1
        return sorted(freq.items(), key=lambda kv: (-kv[1], labels.index(kv[0])))[0][0]

    return pick_majority(down_labels), pick_majority(up_labels)

# =========================================================
# YOLOv8로 차량 카운팅
# =========================================================
def count_left_right_with_centerline(model: YOLO, img_path: str,
                                     center_ids: Set[int], vehicle_ids: Set[int],
                                     conf: float, iou: float, imgsz: int, save_vis: bool = True, vis_dir: str = "counted_out") -> Tuple[int, int]:
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 이미지를 읽을 수 없습니다: {img_path}")
        return 0, 0
    H, W = img.shape[:2]
    fallback_x = W // 2

    res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    # 중앙선 lookup 생성
    center_lookups = []
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.array([])
        for i in range(masks.shape[0]):
            cid = int(clses[i]) if i < clses.shape[0] else -1
            if cid in center_ids:
                m = (masks[i] > 0.5).astype(np.uint8) * 255
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)), iterations=1)
                lu = robust_centerline_lookup(m, min_band_width=3)
                if lu:
                    center_lookups.append(lu)

    center_lookup_raw = merge_lookups(center_lookups) if center_lookups else {}
    center_lookup = smooth_centerline_lookup(center_lookup_raw, med_ks=21, ema_alpha=0.15)
    center_lookup = clamp_dx(center_lookup, max_dx=8)

    left = right = 0


    # --- 시각화용 복사본 ---
    vis = img.copy()

    if res.boxes is not None:
        bxyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        bcls  = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cid in zip(bxyxy, bcls):
            if cid not in vehicle_ids:
                continue
            cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
            x_mid = x_on_centerline(center_lookup, cy, H, fallback_x)
            if cx < x_mid:
                left += 1; color = (255, 120, 120)   # L
            else:
                right += 1; color = (120, 220, 255)  # R
            # 박스/중심점 그리기
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 3, color, -1)

    # 중앙선 그리기
    draw_poly_from_lookup(vis, center_lookup, step=4, color=(0, 165, 255), thickness=4)

    # 저장
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
        out_path = os.path.join(vis_dir, os.path.splitext(os.path.basename(img_path))[0] + "_counted.jpg")
        cv2.imwrite(out_path, vis)

    return left, right



# =========================================================
# CSV 저장: 날짜별 파일, 행=시간(HHMMSS), 열=[하행 한글, 상행 한글]
# =========================================================
def ensure_csv_header(csv_path: str, down_label: str, up_label: str):
    """
    CSV가 없으면 새로 만들고 헤더: time,<down_label>,<up_label>
    이미 있으면, 헤더가 다르면 기존 파일을 보존하고 경고 출력(단순화).
    """
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['time', down_label, up_label])
        return

    # 파일 존재 시, 헤더 확인
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            first = f.readline().strip()
        # 기대 헤더
        expected = f"time,{down_label},{up_label}"
        if first != expected:
            # 헤더가 바뀌면 뒤에 추가 컬럼으로 합치는 등의 로직도 가능하나,
            # 여기선 단순 경고만 하고 기존 헤더 유지.
            print(f"[WARN] 기존 CSV 헤더와 매핑 결과가 다릅니다.\n"
                  f" - 기존: {first}\n - 새로 감지: {expected}\n"
                  f" 같은 파일에 계속 append는 하지만 컬럼이 일치하지 않을 수 있습니다.")
    except Exception as e:
        print(f"[WARN] CSV 헤더 확인 실패: {e}")

def append_csv_row(csv_path: str, time_hhmmss: str, down_val: int, up_val: int):
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow([time_hhmmss, down_val, up_val])

def collect_images(dir_glob: str, img_glob: str) -> list:
    imgs = []
    # 폴더 글롭이 있으면 해당 폴더들 안의 frame_*.jpg 수집
    if dir_glob:
        dirs = sorted(set(glob.glob(dir_glob)))
        for d in dirs:
            if not os.path.isdir(d):
                continue
            imgs.extend(glob.glob(os.path.join(d, "frame_*.jpg")))
    # 개별 이미지 글롭도 함께 지원(겸용 가능)
    if img_glob:
        for p in [p.strip() for p in img_glob.split(",") if p.strip()]:
            imgs.extend(glob.glob(p))
    return sorted(set(imgs))

# =========================================================
# 파일명 파서
# =========================================================
def parse_frame_name(basename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    frame_YYYYMMDD_HHMMSS.jpg → ('YYYYMMDD', 'HHMMSS')
    """
    m = re.match(r'^frame_(\d{8})_(\d{6})', basename)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def aggregate_csv_by_minute(csv_path: str, output_suffix: str = "_per_min") -> str:
    """
    기존 {date}_video.csv (열: time, <한글1>, <한글2>)를 읽어서
    time의 앞 4자리(HHMM) 기준으로 평균을 내고 새 CSV로 저장.
    예) 15:49:45, 15:49:58 ... → time=1549 한 행으로 평균.
    반환: 새로 저장된 CSV 경로
    """
    import pandas as pd
    import os

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV가 없어 분 단위 집계를 건너뜁니다: {csv_path}")
        return ""

    df = pd.read_csv(csv_path, dtype={"time": str})
    # time 6자리 보장
    df["time"] = df["time"].astype(str).str.zfill(6)

    # HHMM 추출
    df["minute"] = df["time"].str.slice(0, 4)

    # 숫자 컬럼만 평균 (time/minute 제외)
    num_cols = [c for c in df.columns if c not in ("time", "minute")]
    if not num_cols:
        print(f"[WARN] 평균 낼 수 있는 수치 컬럼이 없습니다: {csv_path}")
        return ""

    grouped = (
        df.groupby("minute", as_index=False)[num_cols]
          .mean()
    )

    # 정수로 반올림(원하면 소수 유지 가능)
    grouped[num_cols] = grouped[num_cols].round(0).astype(int)

    # 컬럼명 'minute' → 'time' (HHMM 형식)
    grouped = grouped.rename(columns={"minute": "time"}).sort_values("time")

    out_path = os.path.splitext(csv_path)[0] + f"{output_suffix}.csv"
    grouped.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 분 단위 평균 CSV 저장: {out_path}")
    return out_path

# =========================================================
# 메인
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="중앙선 기준 좌/우 차량 카운트 + 화살표/한글 매핑 → CSV")
    ap.add_argument("--img_glob", type=str, default="frame_*.jpg",
                    help='처리할 이미지 글롭 (쉼표로 여러개: "*.jpg,*.png")')

    ap.add_argument("--dir_glob", type=str, default="", help='폴더 글롭 (예: "2025-08-27_*"). 이 폴더들 아래의 frame_*.jpg를 모두 처리')

    # YOLOv8(seg) 설정
    ap.add_argument("--v8_weights", type=str, default="yolov8n-seg.pt", help="YOLOv8 세그 가중치(.pt)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou",  type=float, default=DEFAULT_IOU)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    # YOLOv5(화살표/텍스트) 설정
    ap.add_argument("--v5_dir", type=str, default="yolov5", help="YOLOv5 디렉토리 (detect.py가 있는 폴더)")
    ap.add_argument("--v5_weights", type=str, default=os.path.join("yolov5", "best.pt"), help="YOLOv5 가중치 (Down/Up/text 학습 모델)")
    # 출력
    ap.add_argument("--out_dir", type=str, default="counted_out", help="결과 이미지 저장 폴더(옵션, 이미지 저장 안 해도 됨)")
    args = ap.parse_args()

    # 이미지 목록 수집
    img_list = collect_images(args.dir_glob, args.img_glob)
    if not img_list:
        print(f"[INFO] 이미지가 없습니다. dir_glob={args.dir_glob}, img_glob={args.img_glob}")
        sys.exit(0)



    # YOLOv8 모델 로드 + 클래스 id 자동 식별
    v8_model = YOLO(args.v8_weights)
    names = v8_model.names  # dict: id -> name
    center_ids  = find_class_ids_by_hints(names, CENTERLINE_HINTS)
    vehicle_ids = find_class_ids_by_hints(names, VEHICLE_HINTS)
    if not center_ids:
        print("[WARN] 중앙선 클래스를 자동으로 찾지 못했습니다. 모델 클래스:", names)
    if not vehicle_ids:
        print("[WARN] 차량 클래스를 자동으로 찾지 못했습니다. 모델 클래스:", names)

    # 날짜별(YYYYMMDD)로 CSV 나눠서 작성
    # 날짜별로 '하행/상행' 한글 라벨을 캐시하여, 같은 날짜에서 재사용(한 장 못 잡혀도 이전 라벨 사용)
    date_to_labels: Dict[str, Tuple[str, str]] = {}  # date -> (down_kor, up_kor)

    total_left = total_right = 0

    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in img_list:
        base = os.path.basename(img_path)
        date_str, time_str = parse_frame_name(base)
        if not date_str or not time_str:
            print(f"[SKIP] 파일명 형식이 다릅니다(필요: frame_YYYYMMDD_HHMMSS.*): {base}")
            continue

        # 1) YOLOv8로 좌/우 카운트
        left, right = count_left_right_with_centerline(
            v8_model, img_path, center_ids, vehicle_ids,
            conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            save_vis=True, vis_dir=args.out_dir
        )
        total_left += left; total_right += right

        # 2) YOLOv5 + OCR로 하행/상행 한글 라벨
        down_kor, up_kor = date_to_labels.get(date_str, (None, None))
        if down_kor is None or up_kor is None:
            arrows_data, texts_data = run_yolov5_detect(args.v5_dir, args.v5_weights, img_path)
            d_kor, u_kor = match_direction_labels(arrows_data, texts_data)
            # 매핑 실패 시 기본 라벨 사용(안전)
            down_kor = d_kor or "하행선"
            up_kor   = u_kor or "상행선"
            date_to_labels[date_str] = (down_kor, up_kor)

        # 3) CSV 저장: {YYYYMMDD}_video.csv  (헤더: time,<하행 라벨>,<상행 라벨>)
        csv_name = f"{date_str}_video.csv"
        csv_path = os.path.join(args.out_dir, csv_name)
        ensure_csv_header(csv_path, down_kor, up_kor)
        # LEFT → 하행(Down) 라벨, RIGHT → 상행(Up) 라벨
        append_csv_row(csv_path, time_str, left, right)

        print(f"[OK] {base} → LEFT={left} (→ {down_kor}), RIGHT={right} (→ {up_kor})  "
              f"→ CSV: {csv_name} (time={time_str})")

    for date_str in sorted(set(date_to_labels.keys())):
        csv_path = os.path.join(args.out_dir, f"{date_str}_video.csv")
        aggregate_csv_by_minute(csv_path)  # → {date}_video_per_min.csv 저장

    print(f"\n=== 전체 합계 (모든 이미지) ===  LEFT: {total_left}  |  RIGHT: {total_right}")
    print("완료!")

if __name__ == "__main__":
    main()
