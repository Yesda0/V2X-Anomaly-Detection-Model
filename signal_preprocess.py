import json
import pandas as pd
from pathlib import Path

# ===== 설정 =====
INPUT_JSON = "spat.json"   # 파일명
OUTPUT_DIR  = "."

# ===== 로드 (list[dict] 형태 가정; {"data":[...]}도 지원) =====
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, dict):
    # 흔한 래핑 키 후보
    for key in ("data", "items", "results"):
        if key in raw and isinstance(raw[key], list):
            raw = raw[key]
            break

if not isinstance(raw, list):
    raise ValueError("JSON 최상위가 리스트가 아니에요. 파일 구조를 확인해줘.")

# ===== 필요한 필드 추출 =====
cols_nm = [
    "etLtsgStatNm","etStsgStatNm",
    "wtLtsgStatNm","wtStsgStatNm",
    "neLtsgStatNm","neStsgStatNm",
    "swLtsgStatNm","swStsgStatNm"
]

records = []
year = month = hour = None  # 제목 만들 때 씀

for d in raw:
    # 케이스별 대소문자, 누락 대비 get
    y  = d.get("trsmYear")
    m  = d.get("trsmMt") or d.get("trsmMon")
    h  = d.get("trsmHr") or d.get("trsmHour")
    tm = d.get("trsmTm") or d.get("trsmTM") or d.get("trsmTime")  # "HHMMSS" 문자열

    # 첫 아이템 기준으로 제목 요소 확보
    if year is None and y is not None:
        year, month, hour = int(y), int(m), int(h)

    row = {
        "trsmTm": str(tm).zfill(6) if tm is not None else None
    }
    for k in cols_nm:
        row[k] = d.get(k)
    records.append(row)

df = pd.DataFrame.from_records(records)

# 시간 정렬(가능하면)
try:
    df = df.sort_values("trsmTm")
except Exception:
    pass

# ===== 제목 및 저장 경로 =====
if year is None:
    # 제목 만들 정보가 없으면 파일명 기본값
    title = "spat_signal_states"
else:
    title = f"{year}{str(month).zfill(2)}{str(hour).zfill(2)}"  # 예: 20251009

out_path = Path(OUTPUT_DIR) / f"{title}_signal_states.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

print(f"✅ 저장 완료: {out_path}")
