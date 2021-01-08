
# coding: utf-8

# In[196]:


import pickle as pk
import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm
import time
import math
import os
import gc
import argparse
import logging
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv
from sklearn import preprocessing


# In[197]:


def plot_scatter(df, save_file, recipe_type):
    
    plt.figure(figsize=(50, 25))#建立圖表1
    plt.suptitle('{} Failure day {}'.format(file_name.split('.')[0],failure_date.date()), fontsize=30)
    
    min_speed = df.speed.min()#-0.02*df.speed.mean()
    max_speed = df.speed.max()#+0.02*df.speed.mean()
    min_torque = df.torque.min()#-0.02*df.torque.mean()
    max_torque = df.torque.max()#+0.02*df.torque.mean()
    
    print(min_speed)
    print(max_speed)
    print(min_torque)
    print(max_torque)
    
    ax1=plt.subplot(3,4,1)#在圖表2中建立子圖1
    ax2=plt.subplot(3,4,2)
    ax3=plt.subplot(3,4,3)
    ax4=plt.subplot(3,4,4)
    ax5=plt.subplot(3,4,5)
    ax6=plt.subplot(3,4,6)
    ax7=plt.subplot(3,4,7)
    ax8=plt.subplot(3,4,8)
    ax9=plt.subplot(3,4,9)
    ax10=plt.subplot(3,4,10)
    ax11=plt.subplot(3,4,11)
    ax12=plt.subplot(3,4,12)

    normal_day = df.starttime.min().date()
    plt.sca(ax1)
    plt.scatter(df[df['date']==normal_day].speed,df[df['date']==normal_day].torque,marker='.')
    plt.title('starttime_{}'.format(normal_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)

    normal_day_after_3_day = df.starttime.min().date()+datetime.timedelta(days=3)
    plt.sca(ax2)
    plt.scatter(df[df['date']==normal_day_after_3_day].speed,df[df['date']==normal_day_after_3_day].torque,marker='.')
    plt.title('{}'.format(normal_day_after_3_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)    

    normal_day_after_5_day = df.starttime.min().date()+datetime.timedelta(days=5)
    plt.sca(ax3)
    plt.scatter(df[df['date']==normal_day_after_5_day].speed,df[df['date']==normal_day_after_5_day].torque,marker='.')
    plt.title('{}'.format(normal_day_after_5_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    normal_day_after_7_day = df.starttime.min().date()+datetime.timedelta(days=7)
    plt.sca(ax4)
    plt.scatter(df[df['date']==normal_day_after_7_day].speed,df[df['date']==normal_day_after_7_day].torque,marker='.')
    plt.title('{}'.format(normal_day_after_7_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_before_5_day = failure_date.date()-datetime.timedelta(days=5)
    plt.sca(ax5)
    plt.scatter(df[df['date']==fail_date_before_5_day].speed,df[df['date']==fail_date_before_5_day].torque,marker='.')
    plt.title('{}'.format(fail_date_before_5_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_before_3_day = failure_date.date()-datetime.timedelta(days=3)
    plt.sca(ax6)
    plt.scatter(df[df['date']==fail_date_before_3_day].speed,df[df['date']==fail_date_before_3_day].torque,marker='.')
    plt.title('{}'.format(fail_date_before_3_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_before_1_day = failure_date.date()-datetime.timedelta(days=1)
    plt.sca(ax7)
    plt.scatter(df[df['date']==fail_date_before_1_day].speed,df[df['date']==fail_date_before_1_day].torque,marker='.')
    plt.title('{}'.format(fail_date_before_1_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date = failure_date.date()
    plt.sca(ax8)
    plt.scatter(df[df['date']==fail_date].speed,df[df['date']==fail_date].torque,marker='.')
    plt.title('failure_date_{}'.format(fail_date.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_after_1_day = failure_date.date()+datetime.timedelta(days=1)
    plt.sca(ax9)
    plt.scatter(df[df['date']==fail_date_after_1_day].speed,df[df['date']==fail_date_after_1_day].torque,marker='.')
    plt.title('{}'.format(fail_date_after_1_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_after_3_day = failure_date.date()+datetime.timedelta(days=3)
    plt.sca(ax10)
    plt.scatter(df[df['date']==fail_date_after_3_day].speed,df[df['date']==fail_date_after_3_day].torque,marker='.')
    plt.title('{}'.format(fail_date_after_3_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_after_5_day = failure_date.date()+datetime.timedelta(days=5)
    plt.sca(ax11)
    plt.scatter(df[df['date']==fail_date_after_5_day].speed,df[df['date']==fail_date_after_5_day].torque,marker='.')
    plt.title('{}'.format(fail_date_after_5_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    fail_date_after_7_day = failure_date.date()+datetime.timedelta(days=7)
    plt.sca(ax12)
    plt.scatter(df[df['date']==fail_date_after_7_day].speed,df[df['date']==fail_date_after_7_day].torque,marker='.')
    plt.title('{}'.format(fail_date_after_7_day.strftime('%Y-%m-%d')), fontsize=30)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    
    plt.savefig('{}{}_{}.jpg'.format(save_file, file_name.split('.')[0], recipe_type))
    plt.show()


# In[198]:


pkl = 'CAODF114R_RR_20200224_20200514.pkl'
recipe_type = 'multiple_recipe'


# In[199]:


path = '/home/jackyyen/NEW_robot_data/'
failure_data = pd.read_csv('/home/jackyyen/record_final_v4.csv', encoding= 'unicode_escape')
save_file = './test/'

fr = open(os.path.join(path, pkl), 'rb')
df = pk.load(fr)
fr.close()

file_name = pkl

machine_tool = file_name.split('_')[0]
machine_tool_axis = file_name.split('_')[1]
machine_tool_strattime = file_name.split('_')[2]
machine_tool_endtime = file_name.split('.')[0].split('_')[3]

print('Failure date:'+str(failure_data[(failure_data['Tool ID']==machine_tool) & (failure_data['Axis']==machine_tool_axis)].iloc[0]['From Time']))
failure_date = failure_data[(failure_data['Tool ID']==machine_tool) & (failure_data['Axis']==machine_tool_axis)].iloc[0]['From Time']
failure_date = datetime.datetime.strptime(failure_date, '%Y/%m/%d %H:%M')

numerical_feature = ['speed','torque']
category_feature = ['starttime', 'point', 'step_name', 'date','recipe']

df = df.rename(columns={'STARTTIME':'starttime','RECIPE':'recipe','POINT':'point','SPEED':'speed','TORQUE':'torque','STEP_NAME':'step_name'})#'
df = df.sort_values(by=['starttime','point']).reset_index(drop=True) #??
df['starttime'] = df['starttime'].apply(lambda x: pd.Timestamp(x))  #
df['date'] = df['starttime'].apply(lambda x: x.date())
df['hour'] = df['starttime'].apply(lambda x: x.hour)

file_min_date = datetime.datetime.strptime(str(min(df['date'])), '%Y-%m-%d')
file_max_date = datetime.datetime.strptime(str(max(df['date'])),'%Y-%m-%d')
file_date_len = file_max_date - file_min_date

recipe_data = df['recipe'].value_counts()#.reset_index()

mask = np.nonzero(df.point.diff()<0)[0]
mask = np.concatenate([[0], mask])

seq_list = []
a = 0

for idx in tqdm(range(len(mask)-1)):
    seq = df.iloc[mask[idx]:mask[idx+1], :]
    if seq['step_name'].iloc[0]==0: #
        #seq.loc[:,numerical_feature] = prepare_data(seq.loc[:,numerical_feature])
        seq_list.append(seq)   #.values
    else:
        a+=1
        print(a)
        
print("Length range of the sequence: {} ~ {}".format(np.min(np.diff(mask)), np.max(np.diff(mask))))
#     print("Mean length of the sequence", np.mean(np.diff(mask)))

max_sequence = np.max(np.diff(mask))

df = df.reset_index(drop=True)
keep_cols = []
mask_idx = np.array(df.point.diff() < 0).nonzero()[0]
mask_idx = np.concatenate([[0], mask_idx])

for idx in tqdm(range(len(mask_idx)-1)):
    keep_cols.append(df.loc[mask_idx[idx]:mask_idx[idx + 1], 'speed'])
print(np.concatenate(keep_cols).shape)


# In[212]:


def prepare_AE_sequence(df, cols_to_drop):
    df = df.reset_index(drop=True)
    keep_cols = []
    mask_idx = np.array(df.point.diff() < 0).nonzero()[0]
    mask_idx = np.concatenate([[0], mask_idx])  # Remember to add the first index
    if (df.columns == 'CHAMBER').any():
        df = df.drop('CHAMBER', axis=1)

    tmp_df = df.drop(cols_to_drop, axis=1)

    scale_seq_list = []
    date_list = []
    raw_list = []
    cor_list = []
    time_list = []
    
    for idx in tqdm(range(len(mask_idx) - 1)):
        
        x1 = df.loc[mask_idx[idx]:mask_idx[idx + 1], 'speed']
        x2 = df.loc[mask_idx[idx]:mask_idx[idx + 1], 'torque']
        X = np.stack((x1, x2), axis=0)
        X_transformed = np.transpose(np.matmul(pinv(np.cov(X[:, :])), X - np.mean(X, axis=1).reshape(-1, 1)))
        cor_list.append(X_transformed)
        
        scale_seq = preprocessing.scale(df.loc[mask_idx[idx]:mask_idx[idx + 1], ['speed','torque']])
        scale_seq_list.append(scale_seq)
        
        raw_df = df.loc[mask_idx[idx]:mask_idx[idx + 1],:]
        raw_list.append(raw_df)

        date = df.loc[mask_idx[idx]:mask_idx[idx + 1], ['date']]
        date_list.append(date)
        
        time = df.loc[mask_idx[idx]:mask_idx[idx + 1], ['starttime']]
        time_list.append(time)

    return np.concatenate(date_list), np.concatenate(time_list),  np.concatenate(raw_list) ,np.concatenate(scale_seq_list), np.concatenate(cor_list)


# # train generator figure

# In[272]:


from random import sample
#df = df.reset_index(drop=True)

train_nor_st_date = df.starttime.min().date()
train_nor_ed_date = df.starttime.min().date() + datetime.timedelta(days=5)

trn_mask = (df['date'] >= train_nor_st_date) & (df['date'] <= train_nor_ed_date)
trn_nor_df = df.loc[trn_mask].reset_index(drop=True)

keep_cols = []
mask_idx = np.array(trn_nor_df.point.diff() < 0).nonzero()[0]
mask_idx = np.concatenate([[0], mask_idx])  # Remember to add the first index
if (trn_nor_df.columns == 'CHAMBER').any():
    trn_nor_df = trn_nor_df.drop('CHAMBER', axis=1)

tmp_df = trn_nor_df.drop(category_feature, axis=1)

scale_seq_list = []
date_list = []
raw_list = []
cor_list = []
time_list = []

for idx in tqdm(range(len(mask_idx) - 1)):

    x1 = trn_nor_df.loc[mask_idx[idx]:mask_idx[idx + 1], 'speed']
    x2 = trn_nor_df.loc[mask_idx[idx]:mask_idx[idx + 1], 'torque']
    X = np.stack((x1, x2), axis=0)
    X_transformed = np.transpose(np.matmul(pinv(np.cov(X[:, :])), X - np.mean(X, axis=1).reshape(-1, 1)))
    cor_list.append(X_transformed)
#     scale_seq = preprocessing.scale(df.loc[mask_idx[idx]:mask_idx[idx + 1], ['speed','torque']])
#     scale_seq_list.append(scale_seq)

#     raw_df = df.loc[mask_idx[idx]:mask_idx[idx + 1],:]
#     raw_list.append(raw_df)

#     date = df.loc[mask_idx[idx]:mask_idx[idx + 1], ['date']]
#     date_list.append(date)

#     time = df.loc[mask_idx[idx]:mask_idx[idx + 1], ['starttime']]
#     time_list.append(time)

np.concatenate(cor_list)[:,:1].min()

min_speed = np.concatenate(cor_list)[:,0].min()#trn_nor_df.speed.min()#-0.02*df.speed.mean()
max_speed = np.concatenate(cor_list)[:,0].max()#trn_nor_df.speed.max()#+0.02*df.speed.mean()
min_torque = np.concatenate(cor_list)[:,1].min()# trn_nor_df.torque.min()#-0.02*df.torque.mean()
max_torque = np.concatenate(cor_list)[:,1].max()#trn_nor_df.torque.max()#+0.02*df.torque.mean()

for i in range(10):
    figure = sample(cor_list[:], 10)
    figure = np.concatenate(figure)
    plt.scatter(figure[:,0],figure[:,1])
    plt.title('{}'.format(123), fontsize=10)
    plt.xlabel('speed')
    plt.ylabel('torque')
    plt.xlim(min_speed,max_speed)
    plt.ylim(min_torque,max_torque)
    plt.show()


# In[ ]:


plt.sca(ax9)
plt.scatter(df[df['date']==fail_date_after_1_day].speed,df[df['date']==fail_date_after_1_day].torque,marker='.')
plt.title('{}'.format(fail_date_after_1_day.strftime('%Y-%m-%d')), fontsize=30)
plt.xlabel('speed')
plt.ylabel('torque')
plt.xlim(min_speed,max_speed)
plt.ylim(min_torque,max_torque)


# # test generator figure

# In[ ]:


# train_nor_st_date = df.starttime.min().date()
# train_nor_ed_date = df.starttime.min().date() + datetime.timedelta(days=5)
test_ab_st_date = failure_date.date() - datetime.timedelta(days=14)
test_ab_ed_date = df.starttime.max().date()

test_mask = (df['date'] >= test_ab_st_date) & (df['date'] <= test_ab_ed_date)
test_ab_df = df.loc[test_mask].reset_index(drop=True)

keep_cols = []
mask_idx = np.array(test_ab_df.point.diff() < 0).nonzero()[0]
mask_idx = np.concatenate([[0], mask_idx])  # Remember to add the first index
if (test_ab_df.columns == 'CHAMBER').any():
    test_ab_df = test_ab_df.drop('CHAMBER', axis=1)

test_ab_df = test_ab_df.drop(category_feature, axis=1)

# scale_seq_list = []
# date_list = []
# raw_list = []
# cor_list = []
# time_list = []


# In[241]:


len(cor_list)


# In[213]:


date, starttime, raw, zscore, cor = prepare_AE_sequence(df, cols_to_drop=category_feature)


# In[222]:


date = pd.DataFrame(date, columns=['date'])
date = date.reset_index(drop=True)
starttime = pd.DataFrame(starttime, columns=['starttime'])
starttime = starttime.reset_index(drop=True)
raw = pd.DataFrame(raw, columns=df.columns)
raw = raw.reset_index(drop=True)
zscore = pd.DataFrame(zscore[:,:2], columns=['speed','torque'])
zscore = zscore.reset_index(drop=True)
cor = pd.DataFrame(cor, columns=['speed','torque'])
cor = cor.reset_index(drop=True)


# In[225]:


test = pd.concat([zscore,date,starttime,raw.point],axis=1)


# In[ ]:


from random import sample


# In[228]:


test


# In[207]:


plot_scatter(test, save_file, recipe_type)


# In[206]:


plot_scatter(df, save_file, recipe_type)
recipe_type = 'single_recipe'
if recipe_type == 'single_recipe':
    for i in recipe_data.index:
        if int(i.split("_")[2].split(" ")[0])>0:
            recipe_max_date = datetime.datetime.strptime(str(max(df[df['recipe']==i]['date'].unique())),'%Y-%m-%d')
            recipe_min_date = datetime.datetime.strptime(str(min(df[df['recipe']==i]['date'].unique())),'%Y-%m-%d')
            recipe_date_len = recipe_max_date-recipe_min_date
            #print(recipe_date_len)
            if recipe_date_len == file_date_len:
                sigle_recipe_df = df[df['recipe']== i]
                break
                
plot_scatter(sigle_recipe_df, save_file, recipe_type)


# In[41]:


mask_idx = np.array(df[df.date==failure_date.date()].point.diff() < 0).nonzero()[0]
mask_idx = np.concatenate([[0], mask_idx])


# In[42]:


df = df.reset_index()


# In[43]:


df_list = []
for idx in tqdm(range(len(mask_idx) - 1)):
    print(len(df.loc[mask_idx[idx]:mask_idx[idx + 1]]))
    df_list.append(df.loc[mask_idx[idx]:mask_idx[idx + 1]])


# In[50]:


recipe_count = df.recipe.value_counts()


# In[63]:


normal_day = df.starttime.min().date()


# In[85]:


df[(df['date']==normal_day) & (df['recipe']=='CT125_150 MS-200_-225 U1')].torque[:].abs().var()


# In[ ]:


df[(df['date']==normal_day-datetime.timedelta(days=1)) & (df['recipe']=='CT125_150 MS-200_-225 U1')].torque[:].abs().var()


# In[84]:


df[(df['date']==fail_date) & (df['recipe']=='CT125_150 MS-200_-225 U1')].torque[:].abs().var()


# In[89]:


df.date.unique()[0]


# In[94]:


fail_date


# In[95]:


for i in df.date.unique():
    print(i)
    print(df[(df['date']==i) & (df['recipe']=='CT125_150 MS-200_-225 U1')].torque[:].abs().var())
    print(df[(df['date']==i) & (df['recipe']=='CT125_150 MS-200_-225 U1')].torque[:].abs().mean())


# In[74]:


fail_date


# In[71]:


fail_date = failure_date.date()


# In[60]:


for i in recipe_count.index:
    #df.starttime.min().date()
    df[df['date']==normal_day and].torque
    #df[


# In[61]:


df.starttime.min().date()


# In[36]:


for i in df_list:
    plt.plot(i.torque)
    plt.plot(i.speed)
    plt.show()

