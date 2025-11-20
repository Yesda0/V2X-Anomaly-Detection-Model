import numpy as np
import pandas as pd

print("="*50)
print("Checking train.npy")
print("="*50)
train_data = np.load('../data/train/train.npy')
print('Train.npy shape:', train_data.shape)
print('Contains NaN:', np.isnan(train_data).any())
print('NaN count:', np.isnan(train_data).sum())
print('Total elements:', train_data.size)
print('Data range: min={}, max={}'.format(np.nanmin(train_data), np.nanmax(train_data)))

print("\n" + "="*50)
print("Checking CSV file")
print("="*50)
csv_path = '../data/final_15202_01.csv'
df = pd.read_csv(csv_path)
print('CSV columns:', df.columns.tolist())
print('CSV shape:', df.shape)

sensor_cols = ['lat_deg', 'lon_deg', 'speed_kmh', '원효대교', '숙명여대', 'travelTime', 'speed']
print('\nNaN count per column:')
for col in sensor_cols:
    nan_count = df[col].isnull().sum()
    print(f'  {col}: {nan_count} ({nan_count/len(df)*100:.1f}%)')

print('\nFirst 5 rows of sensor data:')
print(df[sensor_cols].head())

print('\nSensor data statistics:')
print(df[sensor_cols].describe())

print('\nChecking for problematic values:')
for col in sensor_cols:
    has_nan = df[col].isnull().any()
    has_inf = np.isinf(df[col]).any()
    print(f'{col}: NaN={has_nan}, Inf={has_inf}, min={df[col].min()}, max={df[col].max()}')
