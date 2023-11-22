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

from real_time_runner import RTRunner
from simple_transformer_with_state import TF_RNN_Past_State
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

init_grid_np = np.random.uniform(-100.0, 100.0, (cst.GRID_NUM, cst.GRID_NUM))
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
print(Rs_aligned_T_pose)

# the state at the T pose, dq not necessary actually and will not be used either
s_init_T_pose = np.zeros(cst.n_dofs * 2)
s_init_T_pose[2] = 0.85
s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])

from provider import IMUSet



def get_mean_readings_3_sec():
    counter = 0
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

 
def viz_point(x, ind):
    pb_c.resetBasePositionAndOrientation(
        p_vids[ind],
        x,
        [0., 0, 0, 1]
    )


if __name__ == '__main__':
    imu_set = IMUSet()

    ''' Load Character Info Moudle '''
    spec = importlib.util.spec_from_file_location(
        "char_info", "amass_char_info.py")
    char_info = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(char_info)

    pb_c, c1, _, p_vids, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 viz_h_map=VIZ_H_MAP,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=True,
                                                 compare_gt=False)

    model = TF_RNN_Past_State(
        input_channels_imu, output_channels,
        rnn_hid_size=512,
        tf_hid_size=1024, tf_in_dim=256,
        n_heads=16, tf_layers=4,
        dropout=0.0, in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=WITH_ACC_SUM,
    )
    model.load_state_dict(torch.load(model_name))

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


    # use real time runner with online data
    rt_runner = RTRunner(
        c1, model, process_frame, s_init_T_pose,
        map_bound=cst.MAP_BOUND,
        grid_size=cst.GRID_SIZE,
        play_back_gt=False,
        five_sbp=USE_5_SBP,
        with_acc_sum=WITH_ACC_SUM,
        multi_sbp_terrain_and_correction=MULTI_SBP_CORRECTION,
    )
    last_root_pos = s_init_T_pose[:3]     # assume always start from (0,0,0.9)

    print('\tFinish.\nStart estimating poses. Press q to quit')
    
    logs = []
    RB_and_acc_t = get_transformed_current_reading()
    logs.append(RB_and_acc_t)
    # rt_runner.record_raw_imu(RB_and_acc_t)
    if is_recording:
        record_buffer = RB_and_acc_t.reshape(1, -1)
    t = 1
    import tkinter as tk
    from scipy.spatial.transform import Rotation
    XX = [100, 0, 200, 0, 200, 100]
    YY = [300, 150, 150, 450, 450, 0]
        
        
    window = tk.Tk()
    window.title("Frame Display")

    # Create a canvas to display your data
    canvas = tk.Canvas(window, width=400, height=600)
    canvas.pack()

    while imu_set.available():
        RB_and_acc_t = get_transformed_current_reading()
        logs.append(RB_and_acc_t)

        frame = RB_and_acc_t
        rot = frame[:54].reshape((6, 3, 3))
        acc = frame[54:].reshape((6, 3))
        # Display each matrix in the frame
        canvas.delete("all")
        canvas.create_text(
            50,
            10,
            text=str(t),
            font=5
        )
        for i in range(6):
            # Create a Rotation object from the rotation matrix
            r = Rotation.from_matrix(rot[i])

            # Convert the rotation to Euler angles with 'XYZ' order
            euler_angles = np.degrees(r.as_euler('xyz'))

            for col in range(3):
                for row in range(3):
                    value = rot[i][row][col]
                    canvas.create_text(
                        XX[i] + col * 40 +50,
                        YY[i] + row * 20 +30,
                        text="{:.2f}".format(value),
                        font=5
                    )
                canvas.create_text(
                    XX[i] + col * 40 +50,
                    YY[i] + 3 * 20 +30,
                    text="{}".format(round(euler_angles[col])),
                    font=5
                )
                canvas.create_text(
                    XX[i] + col * 40 +50,
                    YY[i] + 4 * 20 +30,
                    text="{:.2f}".format(acc[i, col]),
                    font=5
                )
            
        window.update()
        # t does not matter, not used
        res = rt_runner.step(RB_and_acc_t, last_root_pos, s_gt=None, c_gt=None, t=t)

        last_root_pos = res['qdq'][:3]

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)

        if t % 15 == 0 and h_id is not None:
            # TODO: double for loop...
            for ii in range(init_grid_np.shape[0]):
                for jj in range(init_grid_np.shape[1]):
                    init_grid_list[jj * init_grid_np.shape[0] + ii] = \
                        rt_runner.region_height_list[rt_runner.height_region_map[ii, jj]]
            h_id, h_b_id = update_height_field_pb(
                pb_c,
                h_data=init_grid_list,
                scale=cst.GRID_SIZE,
                terrainShape=h_id,
                terrain=h_b_id
            )

        # clock.tick(FREQ)

        sys.stdout.write(f"\r{imu_set.count}/{imu_set.length}")
        sys.stdout.flush()
        t += 1

    torch.save(logs, 'data/logs.pt')
    print('Finish.')
