import importlib.util
import pickle
import socket
import threading
import time, sys
from datetime import datetime
import torch
import numpy as np
from fairmotion.ops import conversions
from pygame.time import Clock

from model import load_runner

from render_funcs import init_viz, update_height_field_pb, COLOR_OURS
# make deterministic
from learning_utils import set_seed
import constants as cst
set_seed(1234567)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_recording = True     # always record imu every 15 sec
record_buffer = None
num_imus = 6

process_frame = 60
FREQ = int(1. / cst.DT)

color = COLOR_OURS

model_name = "output/model-with-dip9and10-cpu.pt"
USE_5_SBP = True
WITH_ACC_SUM = True
MULTI_SBP_CORRECTION = False
VIZ_H_MAP = True
MAX_ACC = 10.0

init_grid_np = np.random.uniform(-10.0, 10.0, (cst.GRID_NUM, cst.GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

input_channels_imu = 6 * (9 + 3)
if USE_5_SBP:
    output_channels = 18 * 6 + 3 + 20
else:
    output_channels = 18 * 6 + 3 + 8

# make an aligned T pose, such that front is x, left is y, and up is z (i.e. without heading)
# the IMU sensor at head will be placed the same way, so we can get the T pose's heading (wrt ENU) easily
# the following are the known bone orientations at such a T pose
Rs_aligned_T_pose = np.array([
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
])

# Rs_aligned_T_pose = np.array([
#     1.0, 0, 0, 0, 1, 0, 0, 0, 1,
#     1.0, 0, 0, 0, 1, 0, 0, 0, 1,
#     1.0, 0, 0, 0, 1, 0, 0, 0, 1,
#     1.0, 0, 0, 0, 1, 0, 0, 0, 1,
#     1.0, 0, 0, 0, 1, 0, 0, 0, 1,
#     1.0, 0, 0, 0, 1, 0, 0, 0, 1,
# ])



Rs_aligned_T_pose = Rs_aligned_T_pose.reshape((6, 3, 3))
Rs_aligned_T_pose = \
    np.einsum('ij,njk->nik', conversions.A2R(np.array([0, 0, np.pi/2])), Rs_aligned_T_pose)
print("Rs_aligned_T_pose:", Rs_aligned_T_pose)

# the state at the T pose, dq not necessary actually and will not be used either
s_init_T_pose = np.zeros(cst.n_dofs * 2)
s_init_T_pose[2] = 0.85
s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])

from provider import IMUSet



def get_mean_readings_3_sec():
    mean_buffer = []
    for i in range(FREQ * 3):
        mean_buffer.append(imu_set.current_reading())

    return np.array(mean_buffer).mean(axis=0)


def get_transformed_current_reading():
    R_and_acc_t = imu_set.current_reading()

    R_Gn_St = R_and_acc_t[: 6*9].reshape((6, 3, 3))
    acc_St = R_and_acc_t[6*9:].reshape((6, 3))

    R_Gp_St = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_St)
    R_Gp_Bt = np.einsum('nij,njk->nik', R_Gp_St, R_B0_S0.transpose((0, 2, 1)))

    acc_Gp = np.einsum('ijk,ik->ij', R_Gp_St, acc_St)
    acc_Gp = acc_Gp - acc_offset_Gp

    acc_Gp = np.clip(acc_Gp, -MAX_ACC, MAX_ACC)

    return np.concatenate((R_Gp_Bt.reshape(-1), acc_Gp.reshape(-1)))

 

if __name__ == '__main__':
    imu_set = IMUSet()

    clock = Clock()
    print('Keep for 3 seconds ...', end='')

    # calibration: heading reset
    R_and_acc_mean = get_mean_readings_3_sec()

    R_Gn_Gp = R_and_acc_mean[:6*9].reshape((6, 3, 3))
    # calibration: acceleration offset
    acc_offset_Gp = R_and_acc_mean[6*9:].reshape(6, 3)      # sensor frame (S) and room frame (Gp) align during this
    print(R_Gn_Gp)

    print('\nWear all imus correctly and press any key.')
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # calibration: bone-to-sensor transform
    R_and_acc_mean = get_mean_readings_3_sec()

    R_Gn_S0 = R_and_acc_mean[: 6 * 9].reshape((6, 3, 3))
    R_Gp_B0 = Rs_aligned_T_pose
    R_Gp_S0 = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_S0)
    R_B0_S0 = np.einsum('nij,njk->nik', R_Gp_B0.transpose((0, 2, 1)), R_Gp_S0)

    print('\tFinish.\nStart estimating poses. Press q to quit')
    
    logs = []
    RB_and_acc_t = get_transformed_current_reading()
    logs.append(RB_and_acc_t)
    
    if is_recording:
        record_buffer = RB_and_acc_t.reshape(1, -1)

    while imu_set.available():
        RB_and_acc_t = get_transformed_current_reading()
        logs.append(RB_and_acc_t)

    torch.save(logs, 'data/logs.pt')
    print('Finish.')
