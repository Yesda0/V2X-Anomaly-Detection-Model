import numpy as np
import matplotlib.pyplot as plt
import os
import cnn_lstm.utils as util

# ===== 수정 1: 데이터 로드 방식 변경 =====
# 기존 인덱스 기반 대신 파일 기반으로 변경

# Load test and reconstructed data
test_data_path = os.path.join(util.test_data_path, "test.npy")
reconstructed_data_path = os.path.join(util.reconstructed_data_path, "test_reconstructed.npy")

test_data = np.load(test_data_path)
test_data = test_data[:, -1, ...]  # only compare the last matrix
reconstructed_data = np.load(reconstructed_data_path)

print("The shape of test data is {}".format(test_data.shape))
print("The shape of reconstructed data is {}".format(reconstructed_data.shape))
# 예상: (샘플 수, 10, 10, 3)

# ===== 수정 2: Valid 데이터 로드 =====
valid_data_path = os.path.join("../data/valid/", "valid.npy")
valid_reconstructed_path = os.path.join("../data/reconstructed/", "valid_reconstructed.npy")

# Valid 데이터가 있는지 확인하고 로드
if os.path.exists(valid_data_path):
    valid_data = np.load(valid_data_path)
    valid_data = valid_data[:, -1, ...]
    print("The shape of valid data is {}".format(valid_data.shape))
    
    # ===== 주의: Valid 데이터도 reconstruct 해야 함 =====
    # convlstm.py에서 valid reconstruction을 추가해야 함
    # 여기서는 valid_reconstructed.npy가 있다고 가정
    if os.path.exists(valid_reconstructed_path):
        valid_reconstructed = np.load(valid_reconstructed_path)
        print("The shape of valid reconstructed data is {}".format(valid_reconstructed.shape))
    else:
        print("Warning: valid_reconstructed.npy not found. Please run validation reconstruction first.")
        # 임시로 더미 데이터 생성 (실제로는 convlstm.py에서 생성해야 함)
        valid_reconstructed = valid_data.copy()
else:
    print("Warning: valid.npy not found. Using test data for threshold computation.")
    # Valid 데이터가 없으면 test 데이터의 일부를 사용
    split_idx = len(test_data) // 3
    valid_data = test_data[:split_idx]
    valid_reconstructed = reconstructed_data[:split_idx]
    test_data = test_data[split_idx:]
    reconstructed_data = reconstructed_data[split_idx:]

# ===== 수정 3: Anomaly score 초기화 (실제 데이터 길이 기반) =====
valid_len = len(valid_data)
test_len = len(test_data)

valid_anomaly_score = np.zeros((valid_len, 1))
test_anomaly_score = np.zeros((test_len, 1))

print(f"Valid samples: {valid_len}, Test samples: {test_len}")

# ===== 수정 4: Validation threshold 계산 (10x10 기준) =====
print("\nComputing validation anomaly scores...")
for i in range(valid_len):
    # 10x10x3 중 첫 번째 채널만 사용 (또는 3개 채널 모두 사용 가능)
    error = np.square(np.subtract(valid_data[i, ..., 0], valid_reconstructed[i, ..., 0]))
    num_anom = len(np.where(error > util.threshold)[0])
    valid_anomaly_score[i] = num_anom

max_valid_anom = np.max(valid_anomaly_score)
threshold = max_valid_anom * util.alpha

print("Max valid anomaly score: %.2f" % max_valid_anom)
print("Threshold (alpha * max): %.2f" % threshold)

# ===== 수정 5: Test anomaly score 계산 =====
print("\nComputing test anomaly scores...")
for i in range(test_len):
    error = np.square(np.subtract(test_data[i, ..., 0], reconstructed_data[i, ..., 0]))
    num_anom = len(np.where(error > threshold)[0])
    test_anomaly_score[i] = num_anom

# ===== 수정 6: 실제 anomaly 정보가 있는 경우에만 처리 =====
anomaly_file = "../data/test_anomaly.csv"
has_anomaly_labels = os.path.exists(anomaly_file)

if has_anomaly_labels:
    print("\nLoading anomaly ground truth...")
    root_cause_gt = np.loadtxt(anomaly_file, delimiter=",", dtype=np.int32)
    
    if len(root_cause_gt.shape) == 1:
        root_cause_gt = root_cause_gt.reshape(1, -1)
    
    num_anomalies = root_cause_gt.shape[0]
    anomaly_pos = root_cause_gt[:, 0]
    anomaly_span = [10, 30, 90]  # 이 값들은 실제 데이터에 맞게 조정
    
    # 위치 조정 (실제 데이터 인덱스에 맞게)
    # 기존 코드의 복잡한 계산을 단순화
    anomaly_pos_adjusted = []
    for i in range(num_anomalies):
        # anomaly_pos[i]를 샘플 인덱스로 변환
        adjusted_pos = anomaly_pos[i] // util.gap_time
        anomaly_pos_adjusted.append(adjusted_pos)
    
    print(f"Found {num_anomalies} anomalies at positions: {anomaly_pos_adjusted}")
else:
    print("\nNo anomaly labels found. Skipping ground truth visualization.")
    has_anomaly_labels = False

# ===== 수정 7: 시각화 =====
print("\nGenerating visualization...")
fig, axes = plt.subplots(figsize=(12, 6))

plt.xticks(fontsize=20)
plt.ylim((0, max(100, np.max(test_anomaly_score) * 1.2)))
plt.yticks(fontsize=20)
plt.plot(test_anomaly_score, 'b', linewidth=2, label='Anomaly Score')

# Threshold line
threshold_line = np.full((test_len), threshold)
axes.plot(threshold_line, color='black', linestyle='--', linewidth=2, label='Threshold')

# Ground truth anomaly spans (if available)
if has_anomaly_labels:
    for k in range(num_anomalies):
        span = anomaly_span[k % len(anomaly_span)]
        span_samples = span // util.gap_time
        axes.axvspan(anomaly_pos_adjusted[k], 
                    anomaly_pos_adjusted[k] + span_samples, 
                    color='red', alpha=0.3, label='Anomaly' if k == 0 else '')

plt.xlabel('Test Sample Index', fontsize=22)
plt.ylabel('Anomaly Score', fontsize=22)
plt.title("MSCRED Anomaly Detection (10 sensors)", fontsize=24)
plt.legend(fontsize=16)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')

fig.tight_layout()

# Save figure
output_dir = "../results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, "anomaly_detection_result.png"), dpi=300)
print(f"Figure saved to {os.path.join(output_dir, 'anomaly_detection_result.png')}")

plt.show()

# ===== 수정 8: Anomaly detection 통계 출력 =====
print("\n" + "="*50)
print("Anomaly Detection Statistics")
print("="*50)
print(f"Total test samples: {test_len}")
print(f"Threshold: {threshold:.2f}")

anomalies_detected = np.where(test_anomaly_score > threshold)[0]
print(f"Number of anomalies detected: {len(anomalies_detected)}")
print(f"Anomaly indices: {anomalies_detected[:20]}...")  # 처음 20개만 출력

if has_anomaly_labels:
    print(f"\nGround truth anomalies: {num_anomalies}")
    print(f"Ground truth positions: {anomaly_pos_adjusted}")
    
    # True Positive 계산
    tp = 0
    for pos in anomaly_pos_adjusted:
        if pos < test_len:
            span = anomaly_span[0] // util.gap_time  # 단순화
            if any(pos <= idx <= pos + span for idx in anomalies_detected):
                tp += 1
    
    print(f"True Positives: {tp}/{num_anomalies}")

# ===== 추가: Anomaly score를 CSV로 저장 =====
score_output_path = os.path.join(output_dir, "anomaly_scores.csv")
np.savetxt(score_output_path, test_anomaly_score, delimiter=",", fmt='%.6f')
print(f"\nAnomaly scores saved to {score_output_path}")

print("\n" + "="*50)
print("Evaluation completed!")
print("="*50)
