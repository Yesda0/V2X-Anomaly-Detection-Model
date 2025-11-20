import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils as util
import numpy as np
import os


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    """ConvLSTM cell implementation for TensorFlow 2.x compatibility"""

    def __init__(self, conv_ndims, input_shape, output_channels, kernel_shape,
                 use_bias=True, skip_connection=False, forget_bias=1.0,
                 initializers=None, name="conv_lstm_cell"):
        super(ConvLSTMCell, self).__init__(name=name)
        self._conv_ndims = conv_ndims
        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._skip_connection = skip_connection
        self._size = tf.TensorShape(input_shape[:2] + [output_channels])
        self._feature_axis = self._conv_ndims

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
        return self._size

    def call(self, inputs, state):
        cell_state, hidden_state = state

        # Concatenate inputs and hidden state
        concat_inputs = tf.concat([inputs, hidden_state], axis=-1)

        # Define kernel shape for gates
        kernel_shape = self._kernel_shape + [concat_inputs.shape[-1], 4 * self._output_channels]

        # Initialize kernels
        kernel = tf.get_variable(
            'kernel', kernel_shape,
            initializer=tf.keras.initializers.glorot_uniform())

        # Convolution for all gates (input, forget, output, candidate)
        gates = tf.nn.conv2d(
            concat_inputs, kernel,
            strides=[1, 1, 1, 1],
            padding='SAME')

        if self._use_bias:
            bias = tf.get_variable(
                'bias', [4 * self._output_channels],
                initializer=tf.zeros_initializer())
            gates = tf.nn.bias_add(gates, bias)

        # Split gates
        input_gate, new_input, forget_gate, output_gate = tf.split(
            gates, 4, axis=-1)

        # Apply activations
        new_cell = tf.nn.sigmoid(forget_gate + self._forget_bias) * cell_state
        new_cell += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)
        output = tf.nn.tanh(new_cell) * tf.nn.sigmoid(output_gate)

        if self._skip_connection:
            output += inputs

        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_cell, output)
        return output, new_state


def cnn_encoder_layer(data, filter_layer, strides):
    """
    CNN encoder layer with SELU activation
    """
    result = tf.nn.conv2d(
        input=data,
        filter=filter_layer,
        strides=strides,
        padding="SAME")
    return tf.nn.selu(result)


def tensor_variable(shape, name):
    """
    Tensor variable declaration initialization
    """
    variable = tf.Variable(tf.zeros(shape), name=name)
    variable = tf.get_variable(name, shape=shape,
                              initializer=tf.keras.initializers.glorot_uniform())
    return variable


def cnn_encoder(data):
    """
    ===== 7x7 입력에 맞게 CNN 구조 설계 =====
    입력: 5 * 7 * 7 * 3 (step_max, n_sensors, n_sensors, win_sizes)

    7x7 구조:
    7x7 -> 7x7 -> 4x4 -> 2x2
    """

    # First layer: 7x7x3 -> 7x7x32
    filter1 = tensor_variable([3, 3, 3, 32], "filter1")
    strides1 = (1, 1, 1, 1)
    cnn1_out = cnn_encoder_layer(data, filter1, strides1)

    # Second layer: 7x7x32 -> 4x4x64
    filter2 = tensor_variable([3, 3, 32, 64], "filter2")
    strides2 = (1, 2, 2, 1)  # stride=2로 크기 절반
    cnn2_out = cnn_encoder_layer(cnn1_out, filter2, strides2)

    # Third layer: 4x4x64 -> 2x2x128
    filter3 = tensor_variable([3, 3, 64, 128], "filter3")
    strides3 = (1, 2, 2, 1)
    cnn3_out = cnn_encoder_layer(cnn2_out, filter3, strides3)

    return cnn1_out, cnn2_out, cnn3_out


def cnn_lstm_attention_layer(input_data, layer_number):
    """
    ConvLSTM layer with attention mechanism
    """
    convlstm_layer = ConvLSTMCell(
        conv_ndims=2,
        input_shape=[input_data.shape[2], input_data.shape[3], input_data.shape[4]],
        output_channels=input_data.shape[-1],
        kernel_shape=[2, 2],
        use_bias=True,
        skip_connection=False,
        forget_bias=1.0,
        initializers=None,
        name="conv_lstm_cell" + str(layer_number))

    outputs, state = tf.nn.dynamic_rnn(convlstm_layer, input_data, dtype=input_data.dtype)

    # Attention mechanism
    attention_w = []
    for k in range(util.step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / util.step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, util.step_max])

    outputs = tf.reshape(outputs[0], [util.step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, input_data.shape[2], input_data.shape[3], input_data.shape[4]])

    return outputs, attention_w


def cnn_decoder_layer(conv_lstm_out_c, filter, output_shape, strides):
    """
    CNN decoder layer with transposed convolution
    """
    deconv = tf.nn.conv2d_transpose(
        value=conv_lstm_out_c,
        filter=filter,
        output_shape=output_shape,
        strides=strides,
        padding="SAME")
    deconv = tf.nn.selu(deconv)
    return deconv


def cnn_decoder(lstm1_out, lstm2_out, lstm3_out):
    """
    ===== 7x7 출력에 맞게 Decoder 설계 =====
    역순으로 복원: 2x2 -> 4x4 -> 7x7
    """

    # 2x2x128 -> 4x4x64
    d_filter3 = tensor_variable([3, 3, 64, 128], "d_filter3")
    dec3 = cnn_decoder_layer(lstm3_out, d_filter3, [1, 4, 4, 64], (1, 2, 2, 1))
    dec3_concat = tf.concat([dec3, lstm2_out], axis=3)

    # 4x4x128 -> 7x7x32
    d_filter2 = tensor_variable([3, 3, 32, 128], "d_filter2")
    dec2 = cnn_decoder_layer(dec3_concat, d_filter2, [1, 7, 7, 32], (1, 2, 2, 1))
    dec2_concat = tf.concat([dec2, lstm1_out], axis=3)

    # 7x7x64 -> 7x7x3
    d_filter1 = tensor_variable([3, 3, 3, 64], "d_filter1")
    dec1 = cnn_decoder_layer(dec2_concat, d_filter1, [1, 7, 7, 3], (1, 1, 1, 1))

    return dec1


def main():
    # ===== 데이터 로드 =====
    matrix_data_path = util.train_data_path + "train.npy"
    matrix_gt_1 = np.load(matrix_data_path)
    
    print("Train data shape:", matrix_gt_1.shape)
    # 예상: (샘플 수, step_max=5, 7, 7, 3)

    sess = tf.Session()

    # ===== placeholder 크기: 7x7 =====
    data_input = tf.compat.v1.placeholder(tf.float32, [util.step_max, 7, 7, 3])

    # CNN encoder
    conv1_out, conv2_out, conv3_out = cnn_encoder(data_input)

    # ===== reshape 크기 조정 (7x7 기준) =====
    conv1_out = tf.reshape(conv1_out, [-1, util.step_max, 7, 7, 32])
    conv2_out = tf.reshape(conv2_out, [-1, util.step_max, 4, 4, 64])
    conv3_out = tf.reshape(conv3_out, [-1, util.step_max, 2, 2, 128])

    # LSTM with attention
    conv1_lstm_attention_out, atten_weight_1 = cnn_lstm_attention_layer(conv1_out, 1)
    conv2_lstm_attention_out, atten_weight_2 = cnn_lstm_attention_layer(conv2_out, 2)
    conv3_lstm_attention_out, atten_weight_3 = cnn_lstm_attention_layer(conv3_out, 3)

    # CNN decoder
    deconv_out = cnn_decoder(conv1_lstm_attention_out, conv2_lstm_attention_out, 
                            conv3_lstm_attention_out)
    
    # Loss function: reconstruction error of last step matrix
    loss = tf.reduce_mean(tf.square(data_input[-1] - deconv_out))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=util.learning_rate).minimize(loss)

    # Variable initialization
    init = tf.global_variables_initializer()
    sess.run(init)

    # ===== 학습 루프 =====
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    num_train_samples = matrix_gt_1.shape[0]
    
    for epoch in range(util.training_iters):
        total_loss = 0
        for idx in range(num_train_samples):
            matrix_gt = matrix_gt_1[idx]
            feed_dict = {data_input: np.asarray(matrix_gt)}
            _, loss_value = sess.run([optimizer, loss], feed_dict)
            total_loss += loss_value
        
        avg_loss = total_loss / num_train_samples
        print(f"Epoch {epoch+1}/{util.training_iters}, Average Loss: {avg_loss:.6f}")
        
        # ===== 모델 저장 =====
        if (epoch + 1) % util.save_model_step == 0:
            saver = tf.train.Saver()
            model_save_path = util.model_path + f"model_epoch_{epoch+1}.ckpt"
            if not os.path.exists(util.model_path):
                os.makedirs(util.model_path)
            saver.save(sess, model_save_path)
            print(f"Model saved at epoch {epoch+1}")

    # ===== Validation 평가 및 Reconstruction =====
    print("\n" + "="*50)
    print("Validating and reconstructing validation data...")
    print("="*50)
    
    valid_data_path = "../data/valid/valid.npy"
    if os.path.exists(valid_data_path):
        matrix_valid = np.load(valid_data_path)
        print("Validation data shape:", matrix_valid.shape)
        
        valid_loss_total = 0
        valid_result_all = []
        
        for idx in range(matrix_valid.shape[0]):
            matrix_gt = matrix_valid[idx]
            feed_dict = {data_input: np.asarray(matrix_gt)}
            result, loss_value = sess.run([deconv_out, loss], feed_dict)
            valid_result_all.append(result)
            valid_loss_total += loss_value
        
        avg_valid_loss = valid_loss_total / matrix_valid.shape[0]
        print(f"Average Validation Loss: {avg_valid_loss:.6f}")
        
        # ===== Valid reconstruction 저장 =====
        valid_reconstructed_path = "../data/reconstructed/"
        if not os.path.exists(valid_reconstructed_path):
            os.makedirs(valid_reconstructed_path)
        
        valid_result_all = np.asarray(valid_result_all).reshape((-1, 10, 10, 3))
        np.save(os.path.join(valid_reconstructed_path, "valid_reconstructed.npy"), 
                valid_result_all)
        print(f"Valid reconstructed data saved: {valid_result_all.shape}")
    else:
        print("Warning: valid.npy not found. Skipping validation.")

    # ===== Test 평가 및 Reconstruction =====
    print("\n" + "="*50)
    print("Testing and reconstructing test data...")
    print("="*50)
    
    matrix_data_path = util.test_data_path + "test.npy"
    matrix_gt_1 = np.load(matrix_data_path)
    print("Test data shape:", matrix_gt_1.shape)
    
    result_all = []
    test_loss_total = 0
    
    num_test_samples = matrix_gt_1.shape[0]
    
    for idx in range(num_test_samples):
        matrix_gt = matrix_gt_1[idx]
        feed_dict = {data_input: np.asarray(matrix_gt)}
        result, loss_value = sess.run([deconv_out, loss], feed_dict)
        result_all.append(result)
        test_loss_total += loss_value
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{num_test_samples} test samples")
    
    avg_test_loss = test_loss_total / num_test_samples
    print(f"Average Test Loss: {avg_test_loss:.6f}")

    # ===== Test reconstruction 저장 (10x10x3) =====
    reconstructed_path = util.reconstructed_data_path
    if not os.path.exists(reconstructed_path):
        os.makedirs(reconstructed_path)
    reconstructed_path = reconstructed_path + "test_reconstructed.npy"

    result_all = np.asarray(result_all).reshape((-1, 10, 10, 3))
    print("Test reconstructed data shape:", result_all.shape)
    np.save(reconstructed_path, result_all)
    
    print("\n" + "="*50)
    print("Training, validation, and testing completed!")
    print("="*50)
    
    # ===== 세션 종료 =====
    sess.close()


if __name__ == '__main__':
    main()
