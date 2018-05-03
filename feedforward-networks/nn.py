from typing import Tuple, Dict

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping


def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    model_wide = Sequential()
    model_wide.add(Dense(units=35, activation='relu', input_dim=n_inputs))
    num_units_wide = [35,35,35]
    for u in num_units_wide:
        model_wide.add(Dense(units=u, activation='relu'))
    model_wide.add(Dense(units=n_outputs, activation='relu'))
    model_wide.compile(optimizer='rmsprop',loss='mse')

    model_deep = Sequential()
    model_deep.add(Dense(units=30, activation='relu', input_dim=n_inputs))
    num_units_deep = [20, 20, 20, 16, 16, 20, 20, 20, 30]
    for u in num_units_deep:
        model_deep.add(Dense(units=u, activation='relu'))
    model_deep.add(Dense(units=n_outputs, activation='relu'))
    model_deep.compile(optimizer='rmsprop',loss='mse')

    return (model_deep, model_wide)


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    model_relu = Sequential()
    model_relu.add(Dense(units=70, activation='relu', input_dim=n_inputs))
    model_relu.add(Dense(units=30, activation='relu'))
    model_relu.add(Dense(units=n_outputs, activation='sigmoid'))
    model_relu.compile(optimizer='sgd', loss= 'binary_crossentropy')

    model_tanh = Sequential()
    model_tanh.add(Dense(units=70, activation='tanh', input_dim=n_inputs))
    model_tanh.add(Dense(units=30, activation='tanh'))
    model_tanh.add(Dense(units=n_outputs, activation='sigmoid'))
    model_tanh.compile(optimizer='sgd', loss= 'binary_crossentropy')

    return (model_relu, model_tanh)


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    model_dropout = Sequential()
    model_dropout.add(Dense(units=100, activation='tanh', input_dim=n_inputs))
    model_dropout.add(Dense(units=50, activation='tanh'))
    model_dropout.add(Dropout(0.1))
    model_dropout.add(Dense(units=n_outputs, activation='softmax'))
    model_dropout.compile(optimizer='adam', loss='categorical_crossentropy')

    model_no_dropout = Sequential()
    model_no_dropout.add(Dense(units=100, activation='tanh', input_dim=n_inputs))
    model_no_dropout.add(Dense(units=50, activation='tanh'))
    model_no_dropout.add(Dense(units=n_outputs, activation='softmax'))
    model_no_dropout.compile(optimizer='adam', loss='categorical_crossentropy')

    return (model_dropout, model_no_dropout)


def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Dict, Model, Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    model_earlystopping = Sequential()
    model_earlystopping.add(Dense(units=100, activation='tanh', input_dim=n_inputs))
    model_earlystopping.add(Dense(units=50, activation='tanh'))
    model_earlystopping.add(Dense(units=n_outputs, activation='sigmoid'))
    model_earlystopping.compile(optimizer='adam', loss='binary_crossentropy')

    model_no_earlystopping = Sequential()
    model_no_earlystopping.add(Dense(units=100, activation='tanh', input_dim=n_inputs))
    model_no_earlystopping.add(Dense(units=50, activation='tanh'))
    model_no_earlystopping.add(Dense(units=n_outputs, activation='sigmoid'))
    model_no_earlystopping.compile(optimizer='adam', loss='binary_crossentropy')

    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

    return (model_earlystopping, dict(callbacks=[earlyStopping]), model_no_earlystopping, dict())