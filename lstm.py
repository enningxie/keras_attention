from keras.layers import LSTM, multiply, Input, Dense, Permute, Reshape, Lambda, RepeatVector, Flatten
from keras.models import Model
from keras import optimizers, losses, metrics
import keras.backend as K
import numpy as np

from utils import get_activations, get_data_recurrtent

INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = True


def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


if __name__ == '__main__':
    N = 30000
    train_x, train_y = get_data_recurrtent(N, TIME_STEPS, INPUT_DIM)

    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(
        optimizer=optimizers.Adam(),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy]
    )

    print(m.summary())

    m.fit([train_x], train_y, epochs=5, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        test_x, test_y = get_data_recurrtent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m, test_x, print_shape_only=True, layer_name='attention_vec')[0],
                                   axis=2).squeeze()
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()