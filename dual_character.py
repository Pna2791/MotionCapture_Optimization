import numpy as np

from bullet_agent import SimAgent
from typing import Union, Tuple
import constants as cst
import pickle, time

from data_utils import (
    viz_current_frame_and_store_fk_info_include_fixed,
    our_pose_2_bullet_format
)


def viz_2_trajs_and_return_fk_records_with_sbp(
        char1: SimAgent,
        char2: SimAgent,
        traj1: np.ndarray,
        traj2: np.ndarray,
        start_t: int,
        end_t: int,
        gui: bool,
        seq_c_viz: Union[np.ndarray, None],
):

    print("==="*30)
    m_len = len(traj1)      # use first length if mismatch

    pq_g_1_s = []
    pq_g_2_s = []

    print("==="*30)
    for t in range(0, 5000):
        pq_g_2 = viz_current_frame_and_store_fk_info_include_fixed(char2, traj2[t])
        pq_g_1 = viz_current_frame_and_store_fk_info_include_fixed(char1, traj1[t])   # GT in grey

        pq_g_1_s.append(pq_g_1)
        pq_g_2_s.append(pq_g_2)

        if seq_c_viz is not None:
            cur_c_viz = seq_c_viz[t, :, :]
            for sbp_i in range(cur_c_viz.shape[0]):
                viz_point(cur_c_viz[sbp_i, :], sbp_i)

        time.sleep(1)


    return traj1[start_t: m_len-end_t], traj2[start_t: m_len-end_t], np.array(pq_g_1_s), np.array(pq_g_2_s)


def post_processing_our_model(
        char: SimAgent,
        ours_out: np.ndarray) -> np.ndarray:
    poses_post = []
    for pose in ours_out:
        pose_post = our_pose_2_bullet_format(char, pose)
        poses_post.append(pose_post.tolist())
    poses_post = np.array(poses_post)

    return poses_post


def viz_point(x, ind):
    pb_client.resetBasePositionAndOrientation(
        VIDs[ind],
        x,
        [0., 0, 0, 1]
    )


from render_funcs import init_viz
import importlib.util

''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location("char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)


init_grid_np = np.random.uniform(-10.0, 10.0, (cst.GRID_NUM, cst.GRID_NUM))
init_grid_list = list(init_grid_np.flatten())
pb_client, c1, c2, VIDs, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=True,
                                                 compare_gt=True)

test_file = 'data/preprocessed_DIP_IMU_v1/dipimu_s_03_01.pkl'
data = pickle.load(open(test_file, "rb"))

frames = 5000
Y = data['nimble_qdq']
Y[:, 6:] = 0

gt_list = Y.copy()
gt_list[:, 1] += 0
# gt_list = gt_list[2000:]

ours_list = Y
ours_list[:, 1] -= 10
# ours_list = ours_list[200:]

traj_1 = post_processing_our_model(c1, gt_list)
traj_2 = post_processing_our_model(c1, ours_list)


print("==="*30)
res_tuple = viz_2_trajs_and_return_fk_records_with_sbp(
    c2, c1, traj_1, traj_2, 30, 6, True, None)      # first 0.5s uninteresting

