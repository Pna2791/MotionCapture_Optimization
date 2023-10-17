import pickle
from typing import Dict, Union, Tuple

import numpy as np
import torch
from torch import nn
from fairmotion.ops import conversions
import time

import constants as cst


def load_model(name):
    from simple_transformer_with_state import TF_RNN_Past_State
    input_channels_imu = 6 * (9 + 3)
    output_channels = 18 * 6 + 3 + 20

    model = TF_RNN_Past_State(
        input_channels_imu, output_channels,
        rnn_hid_size=512,
        tf_hid_size=1024, tf_in_dim=256,
        n_heads=16, tf_layers=4,
        dropout=0.0, in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=True
    )
    model.load_state_dict(torch.load(name))
    return model

model_name = "output/model-with-dip9and10-cpu.pt"
model = load_model(model_name)
m_len=500
sum_duration = 0
t_start = time.time()
for t in range(41, m_len-1):
    with open(f'input/in_imu_{t}.pkl', 'rb') as file_in_imu, open(f'input/in_s_and_c_{t}.pkl', 'rb') as file_in_s_and_c:
        in_imu = pickle.load(file_in_imu)
        in_s_and_c = pickle.load(file_in_s_and_c)
        x_imu = torch.tensor(in_imu).float().unsqueeze(0)
        x_s_and_c = torch.tensor(in_s_and_c).float().unsqueeze(0)
        t_start = time.time()
        y = model(x_imu, x_s_and_c).cpu()
        duration = time.time() - t_start
        sum_duration+=duration

n_length = m_len-5 #3000-5
print('Duration:', sum_duration) # 236.84416246414185 - core model
print('FPS:', n_length/sum_duration) # 69.357842005 - core model



"""t_start = time.time()
for i in range(0,in_imu.shape[0]):
    x_imu = torch.tensor(in_imu[i]).float().unsqueeze(0)
    x_s_and_c = torch.tensor(in_s_and_c[i]).float().unsqueeze(0)
    print(x_imu.shape,x_s_and_c.shape)
    y = model(x_imu,x_s_and_c).cpu()
    print(y)
print(time.time()-t_start)"""







