"""
Room and signal processing params
"""
import pyroomacoustics as pra
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

windows = [1280, 2560, 5120] #生成语音信号的样本点
window_size = 5120
if window_size == 5120: # 语音信号的样本点数不一样，CNN的全连接层的长度也不一样
    fc_size = 3712
if window_size == 1280:
    fc_size = 896

n_mic = 16 # 麦克风阵列的阵元数量
fs = 16000
c = 343

# Add microphones to 3D room
height_mic = 1.2
mic_center = np.array([1.7, 7.0, 0.96])
R = pra.linear_2D_array(mic_center[:2], M=n_mic, phi=0, d=0.15) # 这只是平面二维阵列函数，生成二维的坐标，论文上的孔径是0.16m，因为有16个麦克风，所以阵列的整体长度是15*0.15m=2.25m
R = np.concatenate((R, np.ones((1, n_mic)) * height_mic), axis=0) # 再连接上一个维度（高度），生成三维的坐标
mics = pra.MicrophoneArray(R, fs).R

SEED = 5000 # 设定种子，方便复现
height_mic = 1.2
nx, ny = 65, 65
X, Y = np.meshgrid(np.linspace(mics[0,:].min()+0.25, mics[0,:].max()-0.25, nx), np.linspace(4.5, 6.5, ny)) # 阵列的整体长度是2.25m，所以两头各缩0.25m，最后得到的网格大小就是1.75m*2m

# 与生成麦克风阵列的位置的方法相似，先确定二维的坐标，然后连接一个维度，形成三维坐标
src_pos = np.array((np.ravel(X), np.ravel(Y))).T # 经过墨西哥草帽函数后，X和Y均为矩阵，ravel函数将矩阵展平为一维数组，再将两个展平后的一维数组 组合成一个二维数组，并转置成n*2维的矩阵
rng = np.random.default_rng(seed=SEED)
height_srcs = rng.uniform(low=1, high=1.5,size=(src_pos.shape[0],)) # 所有声源的高度都在[1m, 1.5m]之间，且服从均匀分布
src_pos = np.concatenate((src_pos, np.expand_dims(height_srcs,axis=-1)),axis=-1) # 在最后一个维度上连接，维度由(n, 2)变成(n, 3)
max_src, min_src = np.max(src_pos), np.min(src_pos) # 将变量展平为一维数据，再取所有数值中的最大或最小值

# 做归一化
src_pos_norm = (src_pos-min_src)/(max_src-min_src)
src_pos_norm = -1 +(2*src_pos_norm)

SEED = 5000

x_min = 1.45
x_max = 1.95
y_min = 5.25
y_max = 5.75
src_pos_test = [] # 储存所有声源位置中下标为idx_delete的那部分，即为测试集
idx_delete = [] # 声源位置集合里面作为测试集的所有元素的下标
for n_s in range(len(src_pos)):
    if src_pos[n_s,0]> x_min and src_pos[n_s,0]<x_max and src_pos[n_s,1] > y_min and src_pos[n_s,1] < y_max:
        src_pos_test.append(src_pos[n_s]) # 加入测试集
        idx_delete.append(n_s)
src_pos = np.delete(src_pos,idx_delete,axis=0) # 删掉，剩下的即使训练集和验证集
src_pos_test = np.array(src_pos_test)
src_pos_train, src_pos_val = train_test_split(src_pos, test_size=0.2, random_state=SEED) # 通过函数将20%的作为验证集，80%作为训练集

# 画出三个集合的所有声源的俯视图（x-y所在的二维平面）
PLOT_SETUP=False
if PLOT_SETUP:
    plt.figure()
    plt.plot(mics[0,:],mics[1,:],'b*')
    plt.plot(src_pos_train[:,0],src_pos_train[:,1],'b*')
    plt.plot(src_pos_val[:,0],src_pos_val[:,1],'r*')
    plt.plot(src_pos_test[:,0],src_pos_test[:,1],'g*')
    plt.axis('equal')
    plt.show()

# audio signals
corpus = pra.datasets.CMUArcticCorpus(download=True) # 下载语音语料库
idx_tracks = rng.choice(np.arange(len(corpus)),size=src_pos_train.shape[0]+src_pos_val.shape[0]+src_pos_test.shape[0]) # 从语料库中随机抽取所需的若干个语音作为声源的语音
# 直接按照索引号从小到大分配给各个集合对应数量的语料库中语音的索引
idx_tracks_train=idx_tracks[:src_pos_train.shape[0]]
idx_tracks_val=idx_tracks[src_pos_train.shape[0]:src_pos_train.shape[0]+src_pos_val.shape[0]]
idx_tracks_test=idx_tracks[src_pos_train.shape[0]+src_pos_val.shape[0]:src_pos_train.shape[0]+src_pos_val.shape[0]+src_pos_test.shape[0]]


