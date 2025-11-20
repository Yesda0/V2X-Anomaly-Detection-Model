# Parameter initialization
gap_time = 10
win_size = [10, 30, 60]
step_max = 5

# ===== 데이터 경로 수정 =====
data_dir = '../data/'  # CSV 파일들이 있는 디렉토리
model_path = '../MSCRED/'
train_data_path = "../data/train/"
test_data_path = "../data/test/"
reconstructed_data_path = "../data/reconstructed/"

# ===== 파일 리스트 정의 =====
# 13개 파일을 train(8개), valid(3개), test(2개)로 분할
train_files = [
    'final_15202_01.csv',
    'final_15202_02.csv',
    'final_15202_03.csv',
    'final_15202_04.csv',
    'final_15202_05.csv',
    'final_15202_06.csv',
    'final_15202_07.csv',
    'final_15202_08.csv'
]

valid_files = [
    'final_15202_09.csv',
    'final_15202_10.csv',
    'final_15202_11.csv'
]

test_files = [
    'final_15202_12.csv',
    'final_15202_13.csv'
]

# ===== 기존 인덱스 기반 파라미터는 제거하거나 주석 처리 =====
# train_start_id = 10  # 더 이상 필요 없음
# train_end_id = 800
# test_start_id = 800
# test_end_id = 2000
# valid_start_id = 800
# valid_end_id = 1000

# Training parameters
training_iters = 50
save_model_step = 1
learning_rate = 0.0002
threshold = 0.005
alpha = 1.5

# Sensor configuration
n_sensor = 7  # 숫자형 센서만 사용할 경우
sensor_columns = ['lat_deg', 'lon_deg', 'speed_km', '원효대교', 
                  '숙명여대', 'travelTime', 'speed']
