import numpy as np
import pyroomacoustics as pra
import tensorflow as tf
import argparse
import os
import gpuRIR
import utils
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)
from tqdm import tqdm
from params import window_size,corpus, idx_tracks_train,idx_tracks_val, src_pos_train, src_pos_val,src_pos_test,idx_tracks_test
from params import mics, n_mic

parser = argparse.ArgumentParser(description='Endtoend data generation')
parser.add_argument('--T60', type=float, help='T60', default=0.1)
parser.add_argument('--SNR', type=int, help='SNR', default=40)
parser.add_argument('--gpu', type=str, help='gpu', default='0')
path = '/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/dataset2'
args = parser.parse_args()
T60 = args.T60
SNR = args.SNR
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch



# Specify room dimensions
room_dim = [3.6, 8.2, 2.4]  # meters
e_absorption, max_order = pra.inverse_sabine(T60, room_dim) # 这两个参数根本没用到，是多余的

for data_split in ['train','val','test']: # 3个集合的数据挨个处理
    print('Computing '+str(data_split) + ' data')

    if data_split == 'train':
        sources_pos = src_pos_train
        corpus_idxs = idx_tracks_train
    if data_split == 'val':
        sources_pos = src_pos_val
        corpus_idxs = idx_tracks_val
    if data_split == 'test':
        sources_pos = src_pos_test
        corpus_idxs = idx_tracks_test

    for j in tqdm(range(len(sources_pos))): # 对于每个声源位置
        signal = corpus[corpus_idxs[j]].data
        fs = corpus[corpus_idxs[j]].fs

        # Convert signal to float 归一化
        signal = signal / (np.max(np.abs(signal)))

        # Compute Signal Correlation Time
        sig_corr_time = utils.compute_correlation_time(signal) # 这个参数实际上不影响训练，但是后面训练完了之后，分析信号的性能时会用到


        # Add source to 3D room (set max_order to a low value for a quick, but less accurate, RIR)
        source_position = sources_pos[j]
        # Add microphones to 3D room


        # att的这两个参数根本没用到，是多余的；fs与48行左右的fs = corpus[corpus_idxs[j]].fs冲撞了，这二者有一个是多余的
        att_diff = 15.0  # Attenuation when start using the diffuse reverberation model [dB]
        att_max = 60.0  # Attenuation at the end of the simulation [dB]
        fs = 16000.0  # Sampling frequency [Hz]

        beta = gpuRIR.beta_SabineEstimation(room_dim, T60)  # Reflection coefficients 房间反射系数
        Tmax = T60
        nb_img = gpuRIR.t2n(Tmax, room_dim)  # Number of image sources in each dimension nb的意思是Number，图像源在每个维度的个数
        RIRs = gpuRIR.simulateRIR(room_dim, beta, np.expand_dims(source_position,1).T, mics.T, nb_img, Tmax, fs)[0] # 传递参数给这个函数，生成RIR

        # 这块内容是将RIR与声源信号卷积得到理想的观测信号（具体操作是先变换到频域做乘积，后做IFFT，即可得到时域信号），后面还要加上高斯白噪声
        fft_len = len(signal) +RIRs.shape[1] -1
        SIG = torch.fft.fft(torch.Tensor(signal),n=fft_len)
        RIRs_fft = torch.fft.fft(torch.tensor(RIRs),n=fft_len, dim=1)
        signal_conv = torch.fft.ifft(torch.multiply(SIG, RIRs_fft),dim=1)

        # AWGN 加性高斯白噪声
        noisy_signal_conv, noise = utils.add_white_gaussian_noise(signal_conv.detach().numpy(), SNR)
        noisy_signal_conv = torch.Tensor(noisy_signal_conv)

        # Split in windows 加窗分帧
        N_wins = int(noisy_signal_conv.shape[-1]/window_size) # 信号长度（样本点数除以窗长）
        frames = torch.reshape(noisy_signal_conv[:,:N_wins*window_size],(n_mic,N_wins,window_size))
        win_sig  = torch.permute(frames, (0,2,1)) # 从这两行代码可以看出，win_sig其实是对noisy_signal_conv做了形状上的改变而已，当然也有少量长度的丢失（因为做了切片操作）

        # Save data
        if data_split =='train' or data_split == 'val':
            train_path = os.path.join(path,data_split) # 训练数据的地址 文件夹
            train_split_path = os.path.join(train_path, 'SNR_' + str(SNR) + '_T60_' + str(T60)) # 训练数据的地址 加上 数据的一点信息 文件夹
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(train_split_path):
                os.makedirs(train_split_path)
            np.savez(file=os.path.join(train_split_path,str(j)), signal=noisy_signal_conv, # 这里的file是文件名，所以str(j)是数据文件的名字，后缀名是.npz
                     src_pos=source_position,
                     win_sig=win_sig)

        if data_split =='test':
            test_path = os.path.join(path,'test')
            test_split_path = os.path.join(test_path, 'SNR_' + str(SNR) + '_T60_' + str(T60))
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            if not os.path.exists(test_split_path):
                os.makedirs(test_split_path)
            np.savez(file=os.path.join(test_split_path, str(j)), signal=noisy_signal_conv,
                     src_pos=source_position,
                     win_sig=win_sig,
                     sig_corr_time=sig_corr_time)



