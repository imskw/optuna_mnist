from keras import layers
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import keras.backend as K
from keras.applications.resnet50 import resnet50

import optuna
import json


class ConvConfig:
    def __init__(self, type, filters=16, kernel_size=3, activation='relu', dropout=0.1, residual=False):
        self.type = type
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.residual = residual
        self.dropout = dropout


def convolution_block(input, config: ConvConfig, max_pooling: bool=False):
    if config.type == 'conv-act-bn':
        net = layers.Conv2D(config.filters, config.kernel_size, padding='same')(input)
        net = layers.Activation(activation=config.activation)(net)
        net = layers.BatchNormalization()(net)
    elif config.type == 'conv-bn-act':
        net = layers.Conv2D(config.filters, config.kernel_size, padding='same')(input)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=config.activation)(net)
    elif config.type == 'act-conv-bn':
        net = layers.Activation(activation=config.activation)(input)
        net = layers.Conv2D(config.filters, config.kernel_size, padding='same')(net)
        net = layers.BatchNormalization()(net)
    elif config.type == 'act-bn-conv':
        net = layers.Activation(activation=config.activation)(input)
        net = layers.BatchNormalization()(net)
        net = layers.Conv2D(config.filters, config.kernel_size, padding='same')(net)
    elif config.type == 'bn-act-conv':
        net = layers.BatchNormalization()(input)
        net = layers.Activation(activation=config.activation)(net)
        net = layers.Conv2D(config.filters, config.kernel_size, padding='same')(net)
    elif config.type == 'bn-conv-act':
        net = layers.BatchNormalization()(input)
        net = layers.Conv2D(config.filters, config.kernel_size, padding='same')(net)
        net = layers.Activation(activation=config.activation)(net)
    else:
        ValueError(type, 'is not defined')

    net = layers.Dropout(config.dropout)(net)

    if config.residual and config.filters == input.shape[-1]:
        net = layers.add([input, net])

    if max_pooling:
        net = layers.MaxPooling2D((2, 2))(net)

    return net


def create_model(n_layers: [int], config_list: [ConvConfig], n_dense_unit: int, dense_act: str):
    input_layer = layers.Input((28, 28, 1))
    net = input_layer
    for n_layer, config in zip(n_layers, config_list):
        for _ in range(n_layer - 1):
            net = convolution_block(net, config)
        net = convolution_block(net, config, True)

    net = layers.Flatten()(net)
    net = layers.Dense(n_dense_unit, activation=dense_act)(net)
    net = layers.Dense(10, activation='softmax')(net)
    model = Model(input_layer, net)
    return model


def create_train_model(trial):
    n_layers = []
    config_list = []

    conv_types = ['conv-act-bn', 'conv-bn-act', 'act-conv-bn', 'act-bn-conv', 'bn-act-conv', 'bn-conv-act']
    activation_types = ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    activation = trial.suggest_categorical('activation', activation_types)
    for i in range(3):
        n_layers.append(trial.suggest_int('block{}/n_layer'.format(i + 1), 1, 10))
        config_list.append(ConvConfig(type=trial.suggest_categorical('block{}/type'.format(i + 1), conv_types),
                                      filters=int(trial.suggest_discrete_uniform('block{}/filters'.format(i + 1), 2, 128, 4)),
                                      kernel_size=int(trial.suggest_discrete_uniform('block{}/kernel_size'.format(i + 1), 3, 9, 2)),
                                      activation=activation,
                                      dropout=trial.suggest_uniform('block{}/dropout'.format(i + 1), 0.0, 1.0),
                                      residual=trial.suggest_categorical('block{}/residual'.format(i + 1), [True, False])
                                      )
                           )
    n_dense_unit = trial.suggest_int('n_dense_unit', 10, 1024)
    model = create_model(n_layers, config_list, n_dense_unit, activation)

    # optimizer
    optimzier_types = ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam']
    optimizer = trial.suggest_categorical('optimizer', optimzier_types)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model


def objective_wrapper():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    train_x = train_x.reshape(-1, 28, 28, 1) / 255
    test_x = test_x.reshape(-1, 28, 28, 1) / 255
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    def objective(trial):
        K.clear_session()
        model = create_train_model(trial)

        history = model.fit(train_x, train_y, epochs=1, verbose=0, batch_size=128, validation_data=(test_x, test_y))
        if history.history["val_acc"][-1] < 0.7:
            return 1 - history.history["val_acc"][-1]

        history = model.fit(train_x, train_y, epochs=9, verbose=0, batch_size=128, validation_data=(test_x, test_y))
        return 1 - max(history.history["val_acc"])

    return objective


def main():
    study = optuna.create_study()
    study.optimize(objective_wrapper(), n_trials=300)
    with open('best_param_{:.4f}.json'.format(study.best_value), 'w') as f:
        json.dump(study.best_params, f, indent=1)
    df = study.trials_dataframe()
    df.to_csv('result.csv')


def create_model_test():
    n_layers = [2, 2]
    config_list = [ConvConfig('conv-bn-act', filters=2, activation='softplus', residual=True),
                   ConvConfig('conv-bn-act', activation='softsign', residual=True)]
    n_dense_unit = 100
    dense_act = 'linear'
    model = create_model(n_layers, config_list, n_dense_unit, dense_act)
    model.compile(loss='categorical_crossentropy', optimizer='nadam')
    model.summary()
    print('ok')


if __name__ == '__main__':
    main()
    # create_model_test()



