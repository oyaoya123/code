import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
def eucliDist(A,B):
    return np.sqrt([sum(np.power((A - B), 2))])

def plot_dist(x, center):
    dist_list = []
    for i in x:
        dist = eucliDist(center, i)
        dist_list.append(dist)
        np_dist = np.concatenate(dist_list)
    #
    # kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    # plt.hist(x1, **kwargs)
    # plt.hist(x2, **kwargs)
    #
    # plt.hist(np_dist)
    # plt.savefig(output_path)
    # plt.show()
    return np_dist

def load_data(path):
    seq = 3000
    stride = 100
    file = os.listdir(path)
    all_df = []
    for i in range(len(file)):
        df = pd.read_csv(os.path.join(path, file[i]))
        for i in range((len(df)-seq)//stride):
            seq_df = df[i*stride:i*stride+seq].values
            #print(seq_df.shape)
            all_df.append(seq_df)

    return np.asarray(all_df)

def minmaxscaler(x, scaler):
    #scaler = preprocessing.MinMaxScaler()
    x1 = x.reshape(x.shape[0]*x.shape[1],x.shape[2]).astype(float)
    #scaler.fit(x1)
    x1 = scaler.transform(x1.astype(float))
    x1 = x1.reshape(x.shape[0],x.shape[1],x.shape[2])
    return x1

def train_minmaxscaler(x):
    scaler = preprocessing.MinMaxScaler()
    x1 = x.reshape(x.shape[0]*x.shape[1],x.shape[2]).astype(float)
    scaler.fit(x1.astype(float))
    x1 = scaler.transform(x1)
    x1 = x1.reshape(x.shape[0],x.shape[1],x.shape[2])
    return x1, scaler