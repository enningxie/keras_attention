import numpy as np
import keras.backend as K
from keras import Sequential
from keras.layers import Dense


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    print('---- activations ----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_colum=1):
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_colum] = y[:, 0]
    return x, y


def get_data_recurrtent(n, time_steps, input_dim, attention_column=10):
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


if __name__ == '__main__':
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(784,), name='input_layer'))
    model.add(Dense(10, activation='softmax', name='output_layer'))
    inp = model.input

    outputs = [layer.output for layer in model.layers]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]
    print(funcs)
