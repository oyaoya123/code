import tensorflow as tf
import numpy as np
from math import ceil
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from tensorflow.keras import backend as K
from .utils import task
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DeepSVDD:
    def __init__(self, keras_model, input_shape=(28, 28, 1), objective='soft-boundary',
                 nu=0.1, representation_dim=32, batch_size=256, lr=1e-4, center=False, center_path='./'):
        self.represetation_dim = representation_dim
        self.objective = objective
        self.keras_model = keras_model
        self.nu = nu
        self.R = tf.get_variable('R', [], dtype=tf.float32, trainable=False) #get_variabel in
        self.c = tf.get_variable('c', [self.represetation_dim], dtype=tf.float32, trainable=False)
        self.warm_up_n_epochs = 10
        self.batch_size = batch_size

        with task('Build graph'):
            self.x = tf.placeholder(tf.float32, [None] + list(input_shape))
            self.latent_op = self.keras_model(self.x)
            self.dist_op = tf.reduce_sum(tf.square(self.latent_op - self.c), axis=-1) #less then square and sum

            if self.objective == 'soft-boundary':
                self.score_op = self.dist_op - self.R ** 2 #+ = ng , - = ok
                penalty = tf.maximum(self.score_op, tf.zeros_like(self.score_op))
                self.loss_op = self.R ** 2 + (1 / self.nu) * penalty

            elif self.objective == 'one-class':
                self.score_op = self.dist_op
                self.loss_op = self.score_op
            else:
                print('wrong objective')
                #break

            opt = tf.train.AdamOptimizer(lr) #Adam(?) paper used sgd!
            self.train_op = opt.minimize(self.loss_op)

        config = tf.ConfigProto()
        self.saver = tf.train.Saver()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, X_test, y_test, output_path, epochs=10, verbose=True):
        self.output_path = output_path
        N = X.shape[0] #data volume
        BS = self.batch_size
        BN = int(ceil(N / BS)) #step per epoch
        chp = None
        self.sess.run(tf.global_variables_initializer())
        c = self._init_c(X) #
        loss_list = []
        with open('{}.pickle'.format(output_path), 'wb') as f:
            pickle.dump(c, f)

        ops = {
            'train': self.train_op, #optimizer
            'loss': tf.reduce_mean(self.loss_op), #compute average
            'dist': self.dist_op,
        }

        K.set_learning_phase(True)

        for i_epoch in range(epochs):
            ind = np.random.permutation(N)
            x_train = X[ind]
            g_batch = tqdm(range(BN)) if verbose else range(BN)

            for i_batch in g_batch:
                x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                results = self.sess.run(ops, feed_dict={self.x: x_batch})  #

                if self.objective == 'soft-boundary' and i_epoch >= self.warm_up_n_epochs:
                    self.sess.run(tf.assign(
                        self.R, self._get_R(results['dist'], self.nu))
                    )
                    #self.sess.close()
            else:
                if verbose:
                    print(K.get_value(results['loss']))
                #else:
            train_results = self.sess.run({'loss': tf.reduce_mean(self.loss_op)}, feed_dict={self.x: X})
            test_results = self.sess.run({'loss': tf.reduce_mean(self.loss_op)}, feed_dict={self.x: X_test})

            loss_list.append([train_results['loss'], test_results['loss']])
            print('{} training loss:'.format(i_epoch) + str(train_results['loss']))
            print('{} validation loss:'.format(i_epoch) + str(test_results['loss']))
            if i_epoch % 10 == 0:
                if chp == None or chp > test_results['loss']:  # test_auc:
                    chp = test_results['loss']  # test_auc
                    #self.keras_model.save_weights("{}_weights.h5".format(output_path))
                    save_path = self.saver.save(self.sess, "{}_weights.ckpt".format(output_path))
                    print("\r {}_{}_weights.ckpt".format(output_path, i_epoch))

            #if i_epoch % 100 ==0:
        plt.plot(loss_list)
        plt.savefig('{}_{}_loss.jpg'.format(output_path, i_epoch))
        plt.close()

    def load_weights(self, weights_path):
        self.saver.restore(self.sess, "{}_weights.ckpt".format(weights_path))
        print("Model restored.")
        
    def predict(self, X):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        scores = list()
        #latent = list()
        #self.saver.restore(self.sess, "{}_weights.ckpt".format(self.output_path))
        #print("Model restored.")
        # Check the values of the variables
        K.set_learning_phase(False)
        #print(self.sess.run({'latent': self.latent_op}, feed_dict={self.x: X}))
        print(self.sess.run({'loss': tf.reduce_mean(self.loss_op)}, feed_dict={self.x: X}))

        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            s_batch = self.sess.run(self.score_op, feed_dict={self.x: x_batch})
            scores.append(s_batch)

        return np.concatenate(scores)#, np.concatenate(latent)

    def _init_c(self, X, eps=1e-1):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        K.set_learning_phase(False)

        with task('1. Get output'):
            latent_sum = np.zeros(self.latent_op.shape[-1])
            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]
                latent_v = self.sess.run(self.latent_op, feed_dict={self.x: x_batch})
                #print(latent_v.shape)
                latent_sum += latent_v.sum(axis=0)

            c = latent_sum / N
            print(c)

        with task('2. Modify eps'):
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

        self.sess.run(tf.assign(self.c, c))

        return self.sess.run(tf.assign(self.c, c))

    def _get_R(self, dist, nu):
        percentile = (1-nu)*100
        #return np.quantile(np.sqrt(dist), 1 - nu)
        return np.percentile(np.sqrt(dist), percentile)
