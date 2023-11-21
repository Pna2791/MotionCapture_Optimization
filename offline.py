# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import pickle
import importlib.util
import random
import time
import argparse
import constants as cst
import onnxruntime as ort
import numpy as np

from torch import nn
from bullet_agent import SimAgent
from real_time_runner_minimal import RTRunnerMin
from typing import Tuple
from render_funcs import init_viz, COLOR_OURS
from learning_utils import set_seed


parser = argparse.ArgumentParser(description='Run our model and related works models')
parser.add_argument('--ours_path_name_kin', type=str, default="output/model-with-dip9and10-cpu-dynamic.onnx",
                    help='')
parser.add_argument('--test_len', type=int, default=30000,
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

args = parser.parse_args()

set_seed(args.seed)

TEST_LEN = args.test_len
RENDER = args.render
MAX_TEST_MOTION_PRE_CAT = 50        
print("RENDER:", RENDER)

USE_5_SBP = args.five_sbp
WITH_ACC_SUM = args.with_acc_sum

MAP_BOUND = cst.MAP_BOUND * 2.0     
GRID_NUM = int(MAP_BOUND/cst.GRID_SIZE) * 2


def run_ours_wrapper_with_c_rt(imu, s_gt, model_name, char) -> (np.ndarray, np.ndarray):
    def load_model(onnx_path):
        providers = [
            ("CUDAExecutionProvider", {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2147483648,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True
            })
            ]
        sess_options = ort.SessionOptions()
        ort_session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        return ort_session

    model_name = args.ours_path_name_kin
    m = load_model(model_name)
    ours_out, c_out, viz_locs_out = test_run_ours_gpt_v4_with_c_rt_minimal(char, s_gt, imu, m, 120)

    return ours_out, c_out, viz_locs_out


def test_run_ours_gpt_v4_with_c_rt_minimal(
        char: SimAgent,
        s_gt: np.array,
        imu: np.array,
        m: nn.Module,
        max_win_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,float]:

    # use real time runner with offline data
    rt_runner = RTRunnerMin(
        char, m, max_win_len, s_gt[0],
        with_acc_sum=WITH_ACC_SUM,
    )

    m_len = imu.shape[0]
    s_traj_pred = np.zeros((m_len, cst.n_dofs * 2))
    s_traj_pred[0] = s_gt[0]

    c_traj_pred = np.zeros((m_len, rt_runner.n_sbps * 4))
    viz_locs_seq = [np.ones((rt_runner.n_sbps, 3)) * 100.0]

    for t in range(0, m_len-1):
        res = rt_runner.step(imu[t, :], s_traj_pred[t, :3])
        s_traj_pred[t + 1, :] = res['qdq']
        c_traj_pred[t + 1, :] = res['ct']

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)
        viz_locs_seq.append(viz_locs)


    # throw away first "trim" predictions (our algorithm gives dummy values)... append dummy value in the end.
    viz_locs_seq = np.array(viz_locs_seq)
    assert len(viz_locs_seq) == len(s_traj_pred)

    # +2 because post-processing moving average filter effectively introduce a bit more delay
    trim = rt_runner.IMU_n_smooth + 2
    s_traj_pred[0:-trim, :] = s_traj_pred[trim:, :]
    s_traj_pred[-trim:, :] = s_traj_pred[-trim-1, :]
    viz_locs_seq[0:-trim, :, :] = viz_locs_seq[trim:, :, :]
    viz_locs_seq[-trim:, :, :] = viz_locs_seq[-trim-1, :, :]

    return s_traj_pred, c_traj_pred, viz_locs_seq

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


init_grid_np = np.random.uniform(-10.0, 10.0, (GRID_NUM, GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

pb_client, c1, c2, VIDs, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=RENDER,
                                                 compare_gt=args.compare_gt,
                                                 color=color,
                                                 viz_h_map=args.viz_terrain)



test_file = 'data/preprocessed_DIP_IMU_v1/dipimu_s_03_01.pkl'
data = pickle.load(open(test_file, "rb"))

frames = 3000
X = data['imu'][:frames]
Y = data['nimble_qdq'][:frames]

# to make all motion equal in stat compute, and run faster
if Y.shape[0] > TEST_LEN:
    rand_start = random.randrange(0, Y.shape[0] - TEST_LEN)
    start = rand_start
    end = rand_start + TEST_LEN
else:
    start = 0
    end = Y.shape[0]
X = X[start: end, :]
Y = Y[start: end, :]

# for clearer visualization, amass data not calibrated well wrt floor
# translation errors are computed from displacement not absolute Y
Y[:, 2] += 0.05       # move motion root 5 cm up


t_start = time.time()
n_length = len(X)
ours, C, ours_c_viz = run_ours_wrapper_with_c_rt(X, Y, args.ours_path_name_kin, c1)
print('Duration:', time.time() - t_start)
print('FPS:', n_length/(time.time()-t_start))
