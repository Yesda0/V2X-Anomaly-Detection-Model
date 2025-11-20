"""
To represent the inter-correlations between different pairs of time series om a multivariate
time  series segment from t-w to t, we construct an n * n signature matrix M based upon the
pairwise inner-product of two time series within the segment.

Construct s (s = 3) signature matrices with different lengths(w = 10, 30, 60)
"""

import numpy as np
import pandas as pd
import utils as util
import os
from sklearn.preprocessing import StandardScaler


class SignatureMatrices:
    def __init__(self):

        print("Loading multiple CSV files...")

        train_data, _, _ = util.load_train_valid_test_data(
                util.train_files,
                util.valid_files,
                util.test_files,
                util.data_dir,
                util.sensor_columns
        )

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)

        
        self.raw_data = train_data.T
        self.series_number = self.raw_data.shape[0]
        self.series_length = self.raw_data.shape[1]
        self.signature_matrices_number = int(self.series_length / util.gap_time)

        print("series_number is", self.series_number)
        print("series_length is", self.series_length)
        print("signature_matrices_number is", self.signature_matrices_number)

    def signature_matrices_generation(self, win):
        """
        Generation signature matrices according win_size and gap_time, the size of raw_data is n * T, n is the number of
        time series, T is the length of time series.
        To represent the inter-correlations between different pairs of time series in a multivariate time series segment
        from t − w to t, we construct an n × n signature matrix Mt based upon the pairwise inner-product of two time series
        within this segment.
        :param win: the length of the time series segment
        :return: the signature matrices
        """

        if win == 0:
            print("The size of win cannot be 0")
            return None

        raw_data = np.asarray(self.raw_data)
        signature_matrices = np.zeros((self.signature_matrices_number, self.series_number, self.series_number))

        for t in range(win, self.series_length, util.gap_time):
            if t // util.gap_time >= self.signature_matrices_number:
                break

            raw_data_t = raw_data[:, t - win:t]
            signature_matrices[t // util.gap_time] = np.dot(raw_data_t, raw_data_t.T) / win

        return signature_matrices

    def generate_train_test(self, signature_matrices):
        """
        Generate train and test dataset, and store them to ../data/train/train.npy and ../data/test/test.npy
        :param signature_matrices:
        :return:
        """
        train_dataset = []

        for data_id in range(util.step_max - 1, self.signature_matrices_number):
            index = data_id - util.step_max + 1

            index_dataset = signature_matrices[:, index:index + util.step_max]
            train_dataset.append(index_dataset)

        train_dataset = np.asarray(train_dataset)
        train_dataset = np.reshape(train_dataset, [-1, util.step_max, self.series_number, self.series_number,
                                                   signature_matrices.shape[0]])

        print("train dataset shape is", train_dataset.shape)

        train_path = util.train_data_path
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        np.save(os.path.join(train_path, "train.npy"), train_dataset)
        
        return train_dataset

class SignatureMatricesValidTest:
    """Valid/Test 데이터를 위한 Signature Matrices 생성"""
    
    def __init__(self, data_files, data_type='valid'):
        print(f"\nLoading {data_type} data...")
        
        # 데이터 로드
        data_list = []
        for file_name in data_files:
            file_path = os.path.join(util.data_dir, file_name)
            df = pd.read_csv(file_path)
            sensor_data = df[util.sensor_columns].values
            data_list.append(sensor_data)
        
        combined_data = np.vstack(data_list)
        
        # 정규화 (Train에서 학습한 scaler 사용해야 함 - 여기서는 단순화)
        scaler = StandardScaler()
        combined_data = scaler.fit_transform(combined_data)
        
        # Transpose
        self.raw_data = combined_data.T
        self.series_number = self.raw_data.shape[0]
        self.series_length = self.raw_data.shape[1]
        self.signature_matrices_number = int(self.series_length / util.gap_time)
        
        print(f"{data_type} - series_number:", self.series_number)
        print(f"{data_type} - series_length:", self.series_length)
    
    def signature_matrices_generation(self, win):
        """동일한 signature matrix 생성 로직"""
        if win == 0:
            return None
        
        raw_data = np.asarray(self.raw_data)
        signature_matrices = np.zeros((self.signature_matrices_number, 
                                      self.series_number, 
                                      self.series_number))
        
        for t in range(win, self.series_length, util.gap_time):
            if t // util.gap_time >= self.signature_matrices_number:
                break
            raw_data_t = raw_data[:, t - win:t]
            signature_matrices[t // util.gap_time] = np.dot(raw_data_t, raw_data_t.T) / win
        
        return signature_matrices
    
    def generate_dataset(self, signature_matrices, save_path, file_name):
        """데이터셋 생성 및 저장"""
        dataset = []
        
        for data_id in range(util.step_max - 1, self.signature_matrices_number):
            index = data_id - util.step_max + 1
            index_dataset = signature_matrices[:, index:index + util.step_max]
            dataset.append(index_dataset)
        
        dataset = np.asarray(dataset)
        dataset = np.reshape(dataset, 
                           [-1, util.step_max, self.series_number, self.series_number,
                            signature_matrices.shape[0]])
        
        print(f"{file_name} dataset shape:", dataset.shape)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, file_name), dataset)
        
        return dataset


if __name__ == '__main__':
    # ===== Train 데이터 처리 =====
    print("="*50)
    print("Processing TRAIN data")
    print("="*50)
    train_matrices = SignatureMatrices()
    train_signature_matrices = []

    for w in util.win_size:
        train_signature_matrices.append(train_matrices.signature_matrices_generation(w))

    train_signature_matrices = np.asarray(train_signature_matrices)
    print("\nTrain signature_matrices shape:", train_signature_matrices.shape)
    # 예상: (3, signature_matrices_number, 7, 7)

    train_matrices.generate_train_test_valid(train_signature_matrices)

    # ===== Valid 데이터 처리 =====
    print("\n" + "="*50)
    print("Processing VALID data")
    print("="*50)
    valid_matrices = SignatureMatricesValidTest(util.valid_files, 'valid')
    valid_signature_matrices = []

    for w in util.win_size:
        valid_signature_matrices.append(valid_matrices.signature_matrices_generation(w))

    valid_signature_matrices = np.asarray(valid_signature_matrices)
    print("\nValid signature_matrices shape:", valid_signature_matrices.shape)

    valid_matrices.generate_dataset(valid_signature_matrices, 
                                    "../data/valid/", 
                                    "valid.npy")

    # ===== Test 데이터 처리 =====
    print("\n" + "="*50)
    print("Processing TEST data")
    print("="*50)
    test_matrices = SignatureMatricesValidTest(util.test_files, 'test')
    test_signature_matrices = []

    for w in util.win_size:
        test_signature_matrices.append(test_matrices.signature_matrices_generation(w))

    test_signature_matrices = np.asarray(test_signature_matrices)
    print("\nTest signature_matrices shape:", test_signature_matrices.shape)

    test_matrices.generate_dataset(test_signature_matrices, 
                                   util.test_data_path, 
                                   "test.npy")

    print("\n" + "="*50)
    print("All signature matrices generated successfully!")
    print("="*50)
