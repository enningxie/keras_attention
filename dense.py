import numpy as np
from utils import get_activations, get_data

np.random.seed(1337)
from keras.models import Model
from keras.layers import Input, Dense, multiply
from keras import optimizers, losses, metrics

input_dim = 32


def build_model():
    inputs = Input(shape=(input_dim,))

    # attention part start
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
    attention_mul = multiply([inputs, attention_probs], name='attention_mul')
    # attebtion part finish

    hidden_layer = Dense(64)(attention_mul)
    output = Dense(1, activation='sigmoid')(hidden_layer)
    model = Model(inputs=[inputs], outputs=output)
    return model


if __name__ == '__main__':
    N = 10000
    inputs_1, outputs = get_data(N, input_dim)

    m = build_model()
    m.compile(
        optimizer=optimizers.Adam(),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy]
    )

    m.fit([inputs_1], outputs, epochs=20, batch_size=64, validation_split=0.5)
    test_x, test_y = get_data(1, input_dim)

    # attention vector corresponds to the second matrix.
    # the first one is the Inputs output.
    attention_vector = get_activations(m, test_x, print_shape_only=True,
                                       layer_name='attention_vec')[0].flatten()
    print('attention = ', attention_vector)

    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')
    plt.show()