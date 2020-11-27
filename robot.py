import pickle as pk
import pandas as pd
import keras
import datetime
import numpy as np
from tqdm import tqdm
import time
import math
from sklearn.model_selection import train_test_split
from dsvdd import deepSVDD, networks
import matplotlib.dates as mdates
import os
import gc
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prepare_AE_sequence(df, cols_to_drop):
    keep_cols = []
    mask = (df.point.diff() < 0)
    mask_idx = np.nonzero(mask)[0]
    mask_idx = np.concatenate([[0], mask_idx])  # Remember to add the first index
    tmp_df = df.drop(cols_to_drop, axis=1)
    seq_list = []
    date_list = []
    print("Length range of the sequence: {} ~ {}".format(np.min(np.diff(mask_idx)), np.max(np.diff(mask_idx))))
    print("Mean length of the sequence", np.mean(np.diff(mask_idx)))
    for idx in tqdm(range(len(mask_idx) - 1)):
        date_list.append([df.loc[mask_idx[idx + 1] - 1, 'date'],  # tmp_df.loc[mask_idx[idx+1]-1, 'hour'],
                          df.loc[mask_idx[idx]:mask_idx[idx + 1], 'speed'].mean(),
                          df.loc[mask_idx[idx]:mask_idx[idx + 1], 'torque'].mean()])
        seq = tmp_df.iloc[mask_idx[idx]:mask_idx[idx + 1], :]
        seq = seq.drop(columns=keep_cols, axis=1)
        seq_list.append(seq)  # .values

    return seq_list, date_list


def plot_result(total_result, Tool_name, Failure_date):
    fig, host = plt.subplots(figsize=(40, 15))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    p1, = host.plot(total_result['date'].unique(), score, "r--", label="score", linewidth=8.0)
    p2, = par1.plot(total_result['date'].unique(), torque, "y--", label="torque", linewidth=4.0)
    p3, = par2.plot(total_result['date'].unique(), speed, "k-", label="speed", linewidth=4.0)
    # p4, = par3.plot(train_test_valid_seq_seqlen[train_test_valid_seq_seqlen['date']==date_time]['seq'],train_test_valid_seq_seqlen[train_test_valid_seq_seqlen['date']==date_time]['torque'],"b-",label="torque", linewidth=4.0)
    plt.axvline(x=Failure_date.date(), ymin=0, ymax=1, color='b', linewidth=8.0)
    plt.axvline(x=Failure_date.date() - datetime.timedelta(days=15), ymin=0, ymax=1, color='g', linewidth=8.0)

    host.set_xlabel("date")
    host.set_ylabel("score")
    par1.set_ylabel("seq_len")
    par2.set_ylabel("speed")
    par3.set_ylabel("torque")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    # par3.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=16, width=2.5, labelsize=50, rotation=45)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    # par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines], fontsize=48, loc='upper right')
    plt.title('{}'.format(Tool_name), fontsize=40)
    # fourth y axis is not shown unless I add this line
    plt.tight_layout()
    plt.savefig('{}/{}.jpg'.format(save_dir,Tool_name))
    #print('###############')
    #plt.show()


try:
    # Parse the arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", '--model_type', help="The model type, default value = 'MLP'", type=str, choices=['MLP', 'LSTM', 'MLP_encoder', 'LSTM_encoder'], default='MLP')
    # parser.add_argument("-p", '--path', help="The path of the dataset, default value = '{}'".format(root_path), type=str, default=root_path)
    parser.add_argument("-t", '--tool_ID', help="The tool ID", type=str, required=True)
    # parser.add_argument("-l", '--max_length', help="The length of the LSTM auto-encoder sequence input, default value = 70", type=int, default=70)
    # parser.add_argument("-mw", '--model_weights', help="The path of the pretrained model weights file, default value = '{}', if the weight file is not existed, train the model from scratch.".format(''), type=str, default='')
    # parser.add_argument("-ed", '--encoding_dimension', help="The encoding dimension of the AE, default value = 1", type=int, default=1)
    args = parser.parse_args()

    # Initialize the hyper-parameters
    #     root_path = args.path
    #     model_type = args.model_type
    #     #toolID = 'CAPIC709R_AXI_2_Z1_20190706_20191004'
    #     toolID = args.tool_ID
    #     MAX_LENGTH = args.max_length
    #     model_weights_file_name = args.model_weights
    toolID = args.tool_ID

except Exception:
    print("Error occurs while parsing arguments! Check the arguments using -h please.")
    print(Exception)
    raise

path = '/home/jackyyen/Sankyo Robot Controller Data/Data/'
failure_data = pd.read_excel('/home/jackyyen/Sankyo Robot Controller Data/Sankyo Robot Maintainence_202009.xlsx')
save_file = './11.27/'
EPOCHS = 1
pkl = toolID
save_dir = save_file+pkl.split('.pkl')[0]

os.mkdir(save_dir)
os.mkdir(save_dir+'/figure')
os.mkdir(save_dir+'/weights')

fr = open(os.path.join(path, pkl), 'rb')
df = pk.load(fr)
fr.close()

file_name = pkl

machine_tool = file_name.split('_')[0]
machine_tool_axis = file_name.split('_')[1]
machine_tool_strattime = file_name.split('_')[2]
machine_tool_endtime = file_name.split('.')[0].split('_')[3]

print('Failure date:'+str(failure_data[(failure_data['Tool ID']==machine_tool) & (failure_data['Start time']==machine_tool_strattime) & (failure_data['End time']==machine_tool_endtime)].iloc[0]['Time']))
failure_date = failure_data[(failure_data['Tool ID']==machine_tool) & (failure_data['Start time']==machine_tool_strattime) & (failure_data['End time']==machine_tool_endtime)].iloc[0]['Time']

failure_date = datetime.datetime.strptime(failure_date,'%Y-%m-%d %H:%M:%S')
numerical_feature = ['speed','torque']
category_feature = ['starttime', 'point', 'step_name', 'date','recipe']

#Preprocessing
df = df.rename(columns={'STARTTIME':'starttime','RECIPE':'recipe','POINT':'point','SPEED':'speed','TORQUE':'torque','STEP_NAME':'step_name'})#'
df = df.sort_values(by=['starttime','point']).reset_index(drop=True) #??
df['starttime'] = df['starttime'].apply(lambda x: pd.Timestamp(x))  #
df['date'] = df['starttime'].apply(lambda x: x.date())


file_min_date = datetime.datetime.strptime(str(min(df['date'])), '%Y-%m-%d')
file_max_date = datetime.datetime.strptime(str(max(df['date'])),'%Y-%m-%d')
file_date_len = file_max_date - file_min_date

recipe_data = df['recipe'].value_counts()#.reset_index()

#
for i in recipe_data.index:
    if int(i.split("_")[2].split(" ")[0])>0:
        recipe_max_date = datetime.datetime.strptime(str(max(df[df['recipe']==i]['date'].unique())),'%Y-%m-%d')
        recipe_min_date = datetime.datetime.strptime(str(min(df[df['recipe']==i]['date'].unique())),'%Y-%m-%d')
        recipe_date_len = recipe_max_date-recipe_min_date
        #print(recipe_date_len)
        if recipe_date_len == file_date_len:
            df = df[df['recipe']== i]
            break

#tmp_df = df.drop(['recipe'],axis=1)
mask = np.nonzero(df.point.diff()<0)[0]
mask = np.concatenate([[0], mask])

seq_list = []
a=0
for idx in tqdm(range(len(mask)-1)):
    seq = df.iloc[mask[idx]:mask[idx+1], :]
    if seq['step_name'].iloc[0]==0: #是否從0加速段開始
        #seq.loc[:,numerical_feature] = prepare_data(seq.loc[:,numerical_feature])
        seq_list.append(seq)   #.values
    else:
        a+=1
        print(a)

#seq = tmp_df.iloc[mask_idx[idx+1]:, :]
#seq_list.append(seq)
#     print("total length of the sequence", len(seq_list))
print("Length range of the sequence: {} ~ {}".format(np.min(np.diff(mask)), np.max(np.diff(mask))))
#     print("Mean length of the sequence", np.mean(np.diff(mask)))

max_sequence = np.max(np.diff(mask))

train_day_number = failure_date.date()-df.starttime.min().date()
train_test_split_date = failure_date.date()-datetime.timedelta(days=14)

test_ab_st_date =  failure_date.date()-datetime.timedelta(days=14)
test_ab_ed_date = df.starttime.max().date()

train_nor_st_date = df.starttime.min().date()
train_nor_ed_date = failure_date.date()-datetime.timedelta(days=15)

print("Train short date: {} ~ {}".format(train_nor_st_date, train_nor_ed_date))
print("Test short date: {} ~ {}".format(test_ab_st_date, test_ab_ed_date))

trn_mask = (df['date'] >= train_nor_st_date) & (df['date'] <= train_nor_ed_date)
trn_nor_df = df.loc[trn_mask].reset_index(drop=True)

test_mask = (df['date'] >= test_ab_st_date) & (df['date'] <= test_ab_ed_date)
test_ab_df = df.loc[test_mask].reset_index(drop=True)

#plt.plot(pd.concat([trn_nor_df.date,test_ab_df.date]).reset_index(drop=True))
#plt.show()
#TODO normalization function #bad
min_max_test = (test_ab_df[numerical_feature] - trn_nor_df[numerical_feature].min())/(trn_nor_df[numerical_feature].max()- trn_nor_df[numerical_feature].min())
min_max_train = (trn_nor_df[numerical_feature] - trn_nor_df[numerical_feature].min())/(trn_nor_df[numerical_feature].max()- trn_nor_df[numerical_feature].min())

test_ab_df.loc[:,numerical_feature] = min_max_test
trn_nor_df.loc[:,numerical_feature] = min_max_train

seq_trn_df, trn_date_list = prepare_AE_sequence(trn_nor_df, cols_to_drop=category_feature)
seq_test_df, test_date_list = prepare_AE_sequence(test_ab_df, cols_to_drop=category_feature)

max_sequence_length = max(len(max(seq_trn_df, key=len)),len(max(seq_test_df, key=len)))

X_train = keras.preprocessing.sequence.pad_sequences(seq_trn_df, maxlen = max_sequence_length, dtype=np.float64, padding='post', truncating='post', value=0.0)
X_test = keras.preprocessing.sequence.pad_sequences(seq_test_df, maxlen = max_sequence_length, dtype=np.float64, padding='post', truncating='post', value=0.0)
X_train_norm, X_valid_norm = train_test_split(X_train, test_size=0.2, random_state=42, shuffle=True)

ae_model, keras_model = networks.CNN1D_AE(X_train.shape)

svdd = deepSVDD.DeepSVDD(
                keras_model,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                representation_dim=16,
                objective='soft-boundary',
                lr=5e-4,
                batch_size=128,
               )

y_valid = np.array([1]*len(X_valid_norm)) #1 is normal

output_path = '{}/weights/{}'.format(save_dir,file_name)

svdd.fit(X_train_norm, X_valid_norm, y_valid, output_path, epochs=EPOCHS,verbose=False)

keras_model.load_weights("{}_weights.h5".format(output_path))

train_result = svdd.predict(X_train_norm)
valid_result = svdd.predict(X_valid_norm)
test_result = svdd.predict(X_test)

#     print(train_result[train_result>0].shape)
#     print(train_result[train_result<0].shape)
#     print(valid_result[valid_result>0].shape)
#     print(valid_result[valid_result<0].shape)
#     print(test_result[test_result>0].shape)
#     print(test_result[test_result<0].shape)

train_valid_test_result = np.concatenate([train_result,valid_result,test_result])
pd.DataFrame(train_valid_test_result).to_csv('{}/{}.csv'.format(save_dir,file_name))
plt.plot(train_valid_test_result)
plt.savefig('{}/figure/each_point_{}.jpg'.format(save_dir,file_name))
#plt.show()

test_date_df = pd.DataFrame(test_date_list)
train_date_df = pd.DataFrame(trn_date_list)
date_plot = test_date_df.rename({0:'date',1:'speed',2:'torque'},axis=1)
date_plot=pd.concat([train_date_df,test_date_df]).rename({0:'date',1:'speed',2:'torque'},axis=1) #全部資料
date_plot=date_plot.reset_index(drop=True)

total_result = pd.concat([pd.DataFrame(np.concatenate([train_result,valid_result,test_result])), date_plot],axis=1)

score = []
speed = []
torque = []
#a = pd.DataFrame(columns=['score','speed','torque'])
for i in total_result['date'].unique():
    #print(total_result[total_result['date']==i].mean())
    score.append(total_result[total_result['date']==i].mean()[0])
    speed.append(total_result[total_result['date']==i].mean()[1])
    torque.append(total_result[total_result['date']==i].mean()[2])

plot_result(total_result, file_name, failure_date)
keras.backend.clear_session()