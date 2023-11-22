import multiprocessing 
import time 
import numpy as np 
import importlib.util
from sensors import IMU
from real_time_runner import RTRunner
from simple_transformer_with_state import TF_RNN_Past_State
import constants as cst
from render_funcs import (
    init_viz, COLOR_OURS, update_height_field_pb, set_color, COLOR_GT
)


model_name = "output/model-with-dip9and10-cpu.pt"
USE_5_SBP = True
WITH_ACC_SUM = True
MULTI_SBP_CORRECTION = False
VIZ_H_MAP = True
RENDER = True
process_frame = 60

MAP_BOUND = cst.MAP_BOUND * 2.0     # some motions are in large range
GRID_NUM = int(MAP_BOUND/cst.GRID_SIZE) * 2

s_init_T_pose = np.zeros(cst.n_dofs * 2)
s_init_T_pose[2] = 0.85
s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])


if __name__ == '__main__': 
    imu_set = IMU()
    imu_set.preprocess()
    
    sensor_process = multiprocessing.Process(target=imu_set.running) 
    sensor_process.start() 
 
    print("Starting") 
    try: 
        
        ''' Load Character Info Moudle '''
        spec = importlib.util.spec_from_file_location(
            "char_info", "amass_char_info.py")
        char_info = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(char_info)
        
        # TODO: really odd, need to be huge for pybullet to work (say. 10.0)
        init_grid_np = np.random.uniform(-10.0, 10.0, (GRID_NUM, GRID_NUM))
        init_grid_list = list(init_grid_np.flatten())

        pb_client, c1, c2, p_vids, h_id, h_b_id = init_viz(
            char_info,
            init_grid_list,
            hmap_scale=cst.GRID_SIZE,
            gui=RENDER,
            compare_gt=False,
            color=COLOR_OURS
        )
        
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
            with_acc_sum=WITH_ACC_SUM,
        )
        
        rt_runner = RTRunner(
            c1, model, process_frame, s_init_T_pose,
            map_bound=cst.MAP_BOUND,
            grid_size=cst.GRID_SIZE,
            play_back_gt=False,
            five_sbp=USE_5_SBP,
            with_acc_sum=WITH_ACC_SUM,
            multi_sbp_terrain_and_correction=MULTI_SBP_CORRECTION,
        )
        
        t = 0
        last_root_pos = s_init_T_pose[:3]
        
        def viz_point(x, ind):
            pb_client.resetBasePositionAndOrientation(
                p_vids[ind],
                x,
                [0., 0, 0, 1]
            )
        while True:
            frame = imu_set.data_queue.get()
            
            res = rt_runner.step(frame, last_root_pos, t=t)

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
                    pb_client,
                    h_data=init_grid_list,
                    scale=cst.GRID_SIZE,
                    terrainShape=h_id,
                    terrain=h_b_id
                )

            t += 1
            
    except Exception as e: 
        print("ERROR:", e) 
    
    sensor_process.kill()
    print("Exited")