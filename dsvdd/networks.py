from tensorflow import keras
from keras.layers import Input, RepeatVector, Dense, Conv1D, LSTM, Reshape, UpSampling1D, Dropout, Activation, CuDNNGRU, Bidirectional, BatchNormalization, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten
from keras.models import Model
from keras.optimizers import nadam, adam
from keras_1d_spp import SPP1DLayer


__all__ = ['CNN1D_spp_AE','CNN1D_AE','LSTM_AE_2','LSTM_AE','mnist_lenet', 'cifar_lenet']

def CNN1D_spp_AE(input_shape):

    inp = Input(shape=(input_shape[1], input_shape[2],))
    conv1 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(inp)
    conv2 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv1)
    #conv3 = Conv1D(64, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv2)
    spp = SPP1DLayer([1,4,16])(conv2)

    #glmaxp1= GlobalMaxPooling1D()(bn1)
    #flatten = Flatten()(spp)
    encoder = Dense(100*16)(spp) #dim
    #fn1 = Dense(5000*16)(encoder)
    reshape = Reshape((100, 16))(encoder)
    #up = UpSampling1D(size=100)(reshape)
    #conv4 = Conv1D(64, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    conv5 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    conv6 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv5)
    outp = Conv1D(input_shape[2], 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv6)
    model = Model(inputs=inp, outputs=outp)
    encoder_model = Model(inputs=inp, outputs=encoder)
    model.compile(loss='mean_squared_error', optimizer=nadam(lr=4*1e-4))
    print(encoder_model.summary())
    print(model.summary())

    return model, encoder_model


def CNN1D_AE(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    conv1 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(inp)
    conv2 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv1)
    #conv3 = Conv1D(64, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv2)

    #glmaxp1= GlobalMaxPooling1D()(bn1)
    flatten = Flatten()(conv2)
    encoder = Dense(16)(flatten) #dim
    fn1 = Dense(5000*16)(encoder)
    reshape = Reshape((5000, 16))(fn1)
    #up = UpSampling1D(size=100)(reshape)
    #conv4 = Conv1D(64, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    conv5 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    conv6 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv5)
    outp = Conv1D(input_shape[2], 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv6)
    model = Model(inputs=inp, outputs=outp)
    encoder_model = Model(inputs=inp, outputs=encoder)
    model.compile(loss='mean_squared_error', optimizer=nadam(lr=4*1e-4))
    print(encoder_model.summary())
    print(model.summary())
    return model, encoder_model

def LSTM_AE(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    #x = Activation('relu')(x)
    x = LSTM(64, kernel_initializer='normal', return_sequences=True)(inp)
    #x = Activation('relu')(x)
    x = LSTM(32, kernel_initializer='normal', return_sequences=False)(x)
    encoder = Dense(16, kernel_initializer='normal', activation='linear')(x)
    #x = Dense(32*100)(encoder)
    #x = Reshape((100, 32))(x)
    x = RepeatVector(input_shape[1])(encoder)
    x = LSTM(32, kernel_initializer='normal', return_sequences=True,)(x)
    x= LSTM(64, kernel_initializer='normal', return_sequences=True)(x)
    outp = LSTM(input_shape[2], kernel_initializer='normal', return_sequences=True)(x)
    #outp = Dense(32, kernel_initializer='normal', activation='linear', use_bias=False)(conc)
    model = Model(inputs=inp, outputs=outp)
    encoder_model = Model(inputs=inp, outputs=encoder)

    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer=adam(lr=4*1e-4))
    return model, encoder_model


def LSTM_AE_2(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    #x = Activation('relu')(x)
    x = LSTM(64, kernel_initializer='normal', return_sequences=True)(inp)
    #x = Activation('relu')(x)
    x = LSTM(32, kernel_initializer='normal', return_sequences=False)(x)
    x = Dense(16, kernel_initializer='normal', activation='linear')(x)
    encoder = Dense(32*100, kernel_initializer='normal', activation='linear')(x)
    #x = Dense(32*100)(encoder)
    x = Reshape((100, 32))(encoder)
    x = LSTM(32, kernel_initializer='normal', return_sequences=True)(x)
    x = LSTM(64, kernel_initializer='normal', return_sequences=True)(x)
    outp = LSTM(input_shape[2], kernel_initializer='normal', return_sequences=True)(x)
    #outp = Dense(32, kernel_initializer='normal', activation='linear', use_bias=False)(conc)
    model = Model(inputs=inp, outputs=outp)
    encoder_model = Model(inputs=inp, outputs=encoder)
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer=adam(lr=4*1e-4))
    return model, encoder_model

def model_LSTM(input_shape):#, MAX_LENGTH):
    inp = Input(shape=(input_shape[1], input_shape[2],))

    x = LSTM(64, kernel_initializer='normal', return_sequences=True, use_bias=False)(inp)
    x = LSTM(32, kernel_initializer='normal', return_sequences=False, use_bias=False)(x)

    outp = Dense(16, kernel_initializer='normal', activation='linear', use_bias=False)(x)
    model = Model(inputs=inp, outputs=outp)

    return model

def mnist_lenet(H=32):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=(28, 28, 1)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False))

    return model


def cifar_lenet(H=128):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same', use_bias=False, input_shape=(32, 32, 3)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(64, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False))

    return model
