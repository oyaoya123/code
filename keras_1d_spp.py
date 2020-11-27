# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:08:27 2020

@author: TW300004580
"""

from keras import layers, models
from keras import backend as K
import numpy as np
import math
import tensorflow as tf

class SPP1DLayer(layers.Layer):
    def __init__(self, splits_list=[1, 4, 16]):
        super(SPP1DLayer, self).__init__(name='spp1dlayer')
        self.splits_list = splits_list
        
    def build(self, input_shape):
        self.Input_shape = input_shape
        
    def compute_output_shape(self, input_shape):
        return (self.Input_shape[0], self.Input_shape[2]*sum([i for i in self.splits_list])) #splits sum(?)
    
    def pooling_operation(self, x, splits):
        inp_shape = K.shape(x)
        remainder = inp_shape[1]%splits
        y = tf.constant(0)
        kernel_size = tf.cond(tf.math.greater(remainder, y),
                              lambda: inp_shape[1]//splits, 
                              lambda: inp_shape[1]//splits+1)
        stride = tf.cond(tf.math.greater(remainder, y),
                              lambda: inp_shape[1]//splits, 
                              lambda: inp_shape[1]//splits+1)
        
        # if K.int_shape(x)[1]%splits == 0:
        #     kernel_size = inp_shape[1]//splits
        #     stride = inp_shape[1]//splits
        # else:
        #     kernel_size = inp_shape[1]//splits
        #     stride = inp_shape[1]//splits
        #kernel_size = inp_shape[1]//splits
        # = inp_shape[1]//splits
        pv = []

        #pooling operation
        for i in range(0, splits): 
            max_value = K.max(x[:,i*stride:i*stride+kernel_size,:],axis=1)
            pv.append(max_value)
        pv = K.concatenate(pv)
        pv = tf.convert_to_tensor(pv)
        return pv    
    
    def call(self, x):
        output = []
        #print(K.shape(x))
        #pooling operation
        for i in self.splits_list:
            pooling = self.pooling_operation(x, i)
            #print(pooling.shape)
            output.append(pooling)
        output = K.concatenate(output)
        print(output.shape)

        return  tf.convert_to_tensor(output)
            
#example
# batch_size=1
# inp = layers.Input(shape=(None, 1))
# #conv= layers.Conv1D(16, 3, activation='relu', padding='same')(inp)
# out = SPP1DLayer([1,4,16])(inp)
# model = models.Model(inputs=inp, outputs=out)
#
# X = np.array(range(23))
# X = X.reshape(1,23,1)
# kvar = K.variable(value=X, dtype='float32', name='example_var')
# #K.print_tensor(model(kvar))
# print(K.eval(model(kvar)))

# model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(np.random.rand(batch_size, 100, 1), np.zeros((batch_size, 336)))
# model.summary()


            