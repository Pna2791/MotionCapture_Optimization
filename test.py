# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import importlib.util
import time
from typing import Union, Tuple

import numpy as np
from torch import nn

from bullet_agent import SimAgent
from real_time_runner import RTRunner

import torch
import os
import argparse

# make deterministic
from data_utils import \
    viz_current_frame_and_store_fk_info_include_fixed, \
    loss_angle, loss_j_pos, loss_root_dist_pos, loss_max_jerk, loss_root_jerk, our_pose_2_bullet_format
from render_funcs import init_viz, COLOR_OURS, update_height_field_pb, set_color, COLOR_GT
from learning_utils import set_seed
import constants as cst



imu_readings_dirs_OUR_format = [
    "syn_HumanEva_v1",
    "preprocessed_DIP_IMU_v1"
]


torch.set_num_threads(1)
np.set_printoptions(threshold=10_000, precision=10)
torch.set_printoptions(threshold=10_000, precision=10)


parser = argparse.ArgumentParser(description='Run our model and related works models')
parser.add_argument('--ours_path_name_kin', type=str, default="output/model-with-dip9and10-cpu.pt",
                    help='')
parser.add_argument('--test_len', type=int, default=3000,
                    help='')
parser.add_argument('--render', default=True, action='store_true',
                    help='')
parser.add_argument('--compare_gt', action='store_true',
                    help='')
parser.add_argument('--seed', type=int, default=42,
                    help='')
parser.add_argument('--five_sbp', default=True, action='store_true',
                    help='')
parser.add_argument('--with_acc_sum', default=True, action='store_true',
                    help='')
parser.add_argument('--viz_terrain', action='store_true',
                    help='')
# parser.add_argument('--save_c', action='store_true',
#                     help='')                # for the DIP-IMU set which has C info
args = parser.parse_args()

set_seed(args.seed)

TEST_LEN = args.test_len
RENDER = args.render
MAX_TEST_MOTION_PRE_CAT = 50        # make testing faster
print("RENDER:", RENDER)

# if args.save_c:
#     MAX_TEST_MOTION_PRE_CAT = 50000
# else:
#     MAX_TEST_MOTION_PRE_CAT = 50
USE_5_SBP = args.five_sbp
WITH_ACC_SUM = args.with_acc_sum

MAP_BOUND = cst.MAP_BOUND * 2.0     # some motions are in large range
GRID_NUM = int(MAP_BOUND/cst.GRID_SIZE) * 2


def run_ours_wrapper_with_c_rt(imu, s_gt, model_name, char) -> (np.ndarray, np.ndarray):
    def load_model(name):
        from simple_transformer_with_state import TF_RNN_Past_State
        input_channels_imu = 6 * (9 + 3)
        if USE_5_SBP:
            output_channels = 18 * 6 + 3 + 20
        else:
            output_channels = 18 * 6 + 3 + 8

        model = TF_RNN_Past_State(
            input_channels_imu, output_channels,
            rnn_hid_size=512,
            tf_hid_size=1024, tf_in_dim=256,
            n_heads=16, tf_layers=4,
            dropout=0.0, in_dropout=0.0,
            past_state_dropout=0.8,
            with_acc_sum=WITH_ACC_SUM
        )
        model.load_state_dict(torch.load(name))
        return model

    m = load_model(model_name)

    ours_out, c_out, viz_locs_out = test_run_ours_gpt_v4_with_c_rt(char, s_gt, imu, m, 40)
    return ours_out, c_out, viz_locs_out


def test_run_ours_gpt_v4_with_c_rt(
        char: SimAgent,
        s_gt: np.array,
        imu: np.array,
        m: nn.Module,
        max_win_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    global h_id, h_b_id

    # use real time runner with offline data
    rt_runner = RTRunner(
        char, m, max_win_len, s_gt,
        map_bound=MAP_BOUND,
        grid_size=cst.GRID_SIZE,
        play_back_gt=False,
        five_sbp=USE_5_SBP,
        with_acc_sum=WITH_ACC_SUM,
        multi_sbp_terrain_and_correction=False
    )

    m_len = imu.shape[0]
    s_traj_pred = np.zeros((m_len, cst.n_dofs * 2))
    c_traj_pred = np.zeros((m_len, rt_runner.n_sbps * 4))
    s_traj_pred[0] = s_gt

    viz_locs_seq = [np.ones((rt_runner.n_sbps, 3)) * 100.0]

    import tkinter as tk
    from scipy.spatial.transform import Rotation
    XX = [100, 0, 200, 0, 200, 100]
    YY = [300, 150, 150, 450, 450, 0]
        
        
    window = tk.Tk()
    window.title("Frame Display")

    # Create a canvas to display your data
    canvas = tk.Canvas(window, width=400, height=600)
    canvas.pack()
    
    for t in range(0, m_len-1):
        
        frame = imu[t, :]
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
        res = rt_runner.step(frame, s_traj_pred[t, :3], t=t)

        s_traj_pred[t + 1, :] = res['qdq']
        c_traj_pred[t + 1, :] = res['ct']

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)

        viz_locs_seq.append(viz_locs)

        if t % 15 == 0 and h_id is not None:
            # TODO: double for loop...
            for ii in range(init_grid_np.shape[0]):
                for jj in range(init_grid_np.shape[1]):
                    init_grid_list[jj*init_grid_np.shape[0]+ii] = \
                        rt_runner.region_height_list[rt_runner.height_region_map[ii, jj]]
            h_id, h_b_id = update_height_field_pb(
                pb_client,
                h_data=init_grid_list,
                scale=cst.GRID_SIZE,
                terrainShape=h_id,
                terrain=h_b_id
            )


def viz_point(x, ind):
    pb_client.resetBasePositionAndOrientation(
        VIDs[ind],
        x,
        [0., 0, 0, 1]
    )


print("[AnhPN]", "==="*30)

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location(
    "char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)

color = COLOR_OURS


# TODO: really odd, need to be huge for pybullet to work (say. 10.0)
init_grid_np = np.random.uniform(-10.0, 10.0, (GRID_NUM, GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

pb_client, c1, c2, VIDs, h_id, h_b_id = init_viz(
    char_info,
    init_grid_list,
    hmap_scale=cst.GRID_SIZE,
    gui=RENDER,
    compare_gt=args.compare_gt,
    color=color,
    viz_h_map=args.viz_terrain
)



X = np.load('output/data.npy')
Y = np.load('output/root.npy')
t_start = time.time()
n_length = len(X)

run_ours_wrapper_with_c_rt(X, Y, args.ours_path_name_kin, c1)

print('Duration:', time.time() - t_start)
print('fps:', n_length/(time.time() - t_start))