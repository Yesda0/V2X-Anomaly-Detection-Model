import os
import glob
import re
import xml.etree.ElementTree as ET
import pandas as pd

# ====== 설정 ======
XML_GLOB = 'its_data_2025-07-25_*.xml'

def extract_file_number(filename: str) -> int:
    m = re.search(r'_(\d+)\.xml', filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0

def to_num(x):
    return pd.to_numeric(x, errors='coerce')

def main():
    xml_files = glob.glob(XML_GLOB)
    if not xml_files:
        print(f"[INFO] XML 파일이 없습니다. 패턴: {XML_GLOB}")
        return

    for xml_file in xml_files:
        print(f"처리 중: {xml_file}")
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                xml_text = f.read().replace('\\','')
            root = ET.fromstring(xml_text)
        except Exception as e:
            print(f"{xml_file} 파싱 실패: {e}")
            continue

        # item별 데이터 수집
        data_dict = {}
        for item in root.findall('.//item'):
            try:
                link_id = (item.findtext('linkId', '') or '').replace(' ','')
                speed = item.findtext('speed', '')
                travel_time = item.findtext('travelTime', '')
                created_date = item.findtext('createdDate', '') or ''

                # YYYYmmddHHMMSS → HHMMSS만 추출
                if len(created_date) >= 14:
                    created_date = created_date[8:14]  # HHMMSS
                # 초 제거 → HHMM
                if len(created_date) == 6:
                    created_date = created_date[:-2]

                if not link_id:
                    continue

                data_dict.setdefault(link_id, []).append({
                    'sourceFile': os.path.basename(xml_file),
                    'createdDate': created_date,   # HHMM
                    'travelTime': travel_time,
                    'speed': speed
                })
            except Exception as e:
                print(f"아이템 처리 오류: {e}")
                continue

        base_name = os.path.basename(xml_file)
        m = re.search(r'(\d{4})[-_\. ]?(\d{2})[-_\. ]?(\d{2})', base_name)
        date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else "unknown"

        # key별 CSV 저장
        for key, rows in data_dict.items():
            try:
                csv_filename = f"{date_str}_{key}.csv"

                if os.path.exists(csv_filename):
                    old_df = pd.read_csv(csv_filename, dtype={'createdDate': str})
                    if 'sourceFile' not in old_df.columns:
                        old_df['sourceFile'] = 'unknown'
                    new_df = pd.DataFrame(rows)
                    new_df['travelTime'] = to_num(new_df['travelTime'])
                    new_df['speed'] = to_num(new_df['speed'])
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                else:
                    combined_df = pd.DataFrame(rows)
                    combined_df['travelTime'] = to_num(combined_df['travelTime'])
                    combined_df['speed'] = to_num(combined_df['speed'])

                # created_dt: 시분만 파싱
                combined_df['created_dt'] = pd.to_datetime(
                    combined_df['createdDate'], format='%H%M', errors='coerce'
                )

                combined_df = combined_df.sort_values(
                    ['created_dt','sourceFile'],
                    ascending=[True,True],
                    kind='mergesort'
                ).drop_duplicates(subset=['created_dt'], keep='last')

                # === 1분 간격 보간 ===
                if len(combined_df) == 0:
                    print(f"{key}: 데이터 없음")
                    continue

                res_df = combined_df[['created_dt','travelTime','speed']].dropna().set_index('created_dt')

                if res_df.shape[0] == 1:
                    out_reset = res_df.reset_index().rename(columns={'created_dt':'createdDate'})
                    out_reset['createdDate'] = out_reset['createdDate'].dt.strftime('%H%M')
                else:
                    start, end = res_df.index.min(), res_df.index.max()
                    grid = pd.date_range(start=start, end=end, freq='T')
                    out = res_df.reindex(grid).sort_index()
                    for col in ['travelTime','speed']:
                        if col in out:
                            out[col] = out[col].interpolate(method='time', limit_direction='both')
                    out_reset = out.reset_index().rename(columns={'index':'createdDate'})
                    out_reset['createdDate'] = out_reset['createdDate'].dt.strftime('%H%M')

                out_reset = out_reset[['createdDate','travelTime','speed']]
                out_reset['travelTime'] = out_reset['travelTime'].round(2)
                out_reset['speed'] = out_reset['speed'].round(2)
 
                out_reset.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                print(f"{csv_filename} 저장 완료 ({len(out_reset)}건)")

            except Exception as e:
                print(f"{key} 처리 오류: {e}")

    print("모든 파일 처리 완료!")

if __name__ == "__main__":
    main()

