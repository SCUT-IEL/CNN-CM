import time
import csv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from device_to_use import *
import numpy as np

window_time = 1
is_16 = False
is_32 = False and (not is_16)
overlap = 0.5
is_se_channel = True  # is channel attention
se_channel_type = 'avg'

lrRate = 1e-3  # learning rate

wav_band = 1
eeg_band = 1
wav_channel = 1
eeg_channel = 64
dataset_name = 'KUL'
if is_16:
    eeg_channel_new = 16
elif is_32:
    eeg_channel_new = 32
else:
    eeg_channel_new = eeg_channel

eeg_s_band = 0
eeg_pool_num = eeg_band
eeg_start = wav_channel + eeg_channel * eeg_s_band
eeg_end = eeg_start + eeg_channel * eeg_band
label = 'B_dot_' + str(window_time) + '_' + dataset_name + '_' + str(eeg_s_band) + 'to' + str(eeg_pool_num)

trail_channel_nor = False
window_nor = False
is_use_wav = False

# data path
oriDataPath = origin_data_document + '/AAD_20_Data_temp/' + dataset_name + '_band' + str(eeg_band)  # matlab后处理文件路径
npyDataPath = './' + dataset_name + '_dataset_cos_' + str(window_time) + 's'  # npy 文件路径

cnnFile = './CNN_base.py'
splitFile = './CNN_split.py'

fs_data = 128
if is_use_wav:
    cnn_conv_hight = eeg_channel_new + 2 * wav_channel
else:
    cnn_conv_hight = eeg_channel_new

window_length = int(fs_data * window_time)  # window length
vali_percent = 0.2
test_percent = 0.2
cnn_ken_num = 10
fcn_input_num = cnn_ken_num * eeg_band

isDS = True  # is both eeg and wav
channel_number = eeg_channel * eeg_band + 2 * wav_channel * wav_band
classLabel = 0

# use gpu
device = torch.device('cuda:' + str(gpu_random))


class mySE(nn.Module):
    def __init__(self, se_weight_num, se_type, se_fcn_squeeze):
        super(mySE, self).__init__()
        se_fcn_num_dict = {'avg': se_weight_num, 'max': se_weight_num}
        se_fcn_num = se_fcn_num_dict.get(se_type)

        self.se_fcn = nn.Sequential(
            nn.Linear(se_fcn_num, se_fcn_squeeze),
            nn.Tanh(),
            nn.Linear(se_fcn_squeeze, se_weight_num),
            nn.Sigmoid(),
            nn.Softmax(dim=1),
        )

    def forward(self, se_data, se_type, eeg_r):
        # 64 channels
        se_weight = eeg_r.view(2, -1)

        se_weight = self.se_fcn(se_weight)

        se_data = torch.cat([se_data, se_data], dim=0)
        se_weight = se_weight.view(1, -1)
        # attention
        output = ((se_data.transpose(0, 2)) * se_weight).transpose(0, 2)

        return output


# the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        if is_se_channel:
            self.se_channel = mySE(eeg_channel_new, se_channel_type, eeg_channel_new)

        self.cnn_conv_eeg = nn.Sequential(
            nn.Conv2d(1, cnn_ken_num, (cnn_conv_hight, 9), stride=(cnn_conv_hight, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((eeg_band * 2, 1)),
        )

        self.cnn_fcn = nn.Sequential(
            nn.Linear(fcn_input_num * 2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
            nn.Softmax(dim=1),
        )

    def dot(selfself, a, b):
        # solve the similarity
        temp = torch.tensordot(a, b, dims=[[0], [0]])
        temp = torch.diag(temp)
        norm_a = torch.norm(a, p=2, dim=0)
        norm_b = torch.norm(b, p=2, dim=0)
        temp = temp / (norm_a * norm_b)
        return temp

    def forward(self, x):
        time_end = str(int(time.time() * 1000))
        # split the eeg and wav data
        eeg = x[:, :, eeg_start:eeg_end, :]
        wav_a = x[:, :, 0:wav_band, :]
        wav_b = x[:, :, -wav_band:, :]

        if is_16:
            index_16 = [0, 33, 39, 37, 4, 14, 12, 47, 49, 51, 57, 30, 20, 26, 28, 63]
            eeg = eeg.view(1, eeg_band, eeg_channel, window_length)
            eeg = eeg[:, :, index_16, :]
            x = torch.cat([wav_a, eeg, wav_b], dim=2)
        elif is_32:
            index_32 = [0, 2, 6, 4, 10, 8, 14, 12, 18, 16, 22, 20, 30, 25, 26, 28, 63, 62, 57, 59, 53, 55, 49, 51, 43,
                        45, 39, 41, 35, 33, 37, 47]
            eeg = eeg.view(1, eeg_band, eeg_channel, window_length)
            eeg = eeg[:, :, index_32, :]
            x = torch.cat([wav_a, eeg, wav_b], dim=2)

        r_a = self.dot(eeg.view(eeg_channel_new, window_length).transpose(0, 1),
                       wav_a.view(1, window_length).transpose(0, 1))
        r_b = self.dot(eeg.view(eeg_channel_new, window_length).transpose(0, 1),
                       wav_b.view(1, window_length).transpose(0, 1))
        eeg_r = torch.cat([r_a, r_b], dim=0).view(2, -1)

        # focus on the important channels
        if is_se_channel:
            eeg = eeg.view(eeg_band, eeg_channel_new, window_length).transpose(0, 1)
            eeg = self.se_channel(eeg, se_channel_type, eeg_r).transpose(0, 1)

        # 卷积过程
        if is_use_wav:
            y = torch.cat([wav_a, eeg, wav_b], dim=2)
            y = y.reshape(1, 1, eeg_band * (eeg_channel_new + 2 * wav_channel), -1)
        else:
            y = eeg
            y = y.reshape(1, 1, eeg_band * eeg_channel_new * 2, -1)

        y = self.cnn_conv_eeg(y)

        y = y.view(1, -1)

        output = self.cnn_fcn(y)

        return output


def weights_init_uniform(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)


# model
myNet = CNN()
myNet.apply(weights_init_uniform)
optimzer = torch.optim.Adam(myNet.parameters(), lr=lrRate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='min', factor=0.1, patience=5, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0,
                                                       eps=0.001)

loss_func = nn.CrossEntropyLoss()
