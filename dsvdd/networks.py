#from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, RepeatVector, Dense, Conv1D, LSTM, Reshape, UpSampling1D, Dropout, Activation, CuDNNGRU, Bidirectional, BatchNormalization, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam, Adam
#from keras_1d_spp import SPP1DLayer
from tensorflow.keras import regularizers

__all__ = ['CNN1D_spp_AE','CNN1D_AE','CNN1D_AE_test','LSTM_AE_2','LSTM_AE','mnist_lenet', 'cifar_lenet']

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
    model.compile(loss='mean_squared_error', optimizer=Nadam(lr=4*1e-4))
    print(encoder_model.summary())
    print(model.summary())

    return model, encoder_model


def CNN1D_AE_test(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    conv1 = tf.keras.layers.Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(inp)
    averagepooling1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, data_format="channels_last")(conv1)
    conv2 = tf.keras.layers.Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(
        averagepooling1)
    flatten = Flatten()(conv2)
    encoder = Dense(16, kernel_regularizer=regularizers.l2(0.01))(flatten)
    fn1 = Dense(conv2.shape[1] * 16)(encoder)
    reshape = Reshape((conv2.shape[1], 16))(fn1)
    conv5 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    unsampling1 = tf.keras.layers.UpSampling1D(size=2)(conv5)
    conv6 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(unsampling1)
    outp = Conv1D(input_shape[2], 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv6)

    model = Model(inputs=inp, outputs=outp)
    encoder_model = Model(inputs=inp, outputs=encoder)
    model.compile(loss='mean_squared_error', optimizer=Nadam(lr=4 * 1e-4))
    # print(encoder_model.summary())
    print(model.summary())
    return model, encoder_model


def CNN1D_AE(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    conv1 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(inp)
    conv2 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv1)
    #conv3 = Conv1D(64, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv2)

    #glmaxp1= GlobalMaxPooling1D()(bn1)
    flatten = Flatten()(conv2)
    #keras.regularizers.l2(l2=0.01, **kwargs)
    encoder = Dense(16, kernel_regularizer=regularizers.l2(0.01))(flatten)#regularizer=regularizers.l2(0.01)) #dim default l2_loss=0.01

    fn1 = Dense(5000*16)(encoder)
    reshape = Reshape((5000, 16))(fn1)
    #up = UpSampling1D(size=100)(reshape)
    #conv4 = Conv1D(64, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    conv5 = Conv1D(16, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(reshape)
    conv6 = Conv1D(8, 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv5)
    outp = Conv1D(input_shape[2], 3, strides=1, padding='same', data_format='channels_last', use_bias=False)(conv6)
    model = Model(inputs=inp, outputs=outp)
    encoder_model = Model(inputs=inp, outputs=encoder)
    model.compile(loss='mean_squared_error', optimizer=Nadam(lr=4*1e-4))
    #print(encoder_model.summary())
    #print(model.summary())
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
