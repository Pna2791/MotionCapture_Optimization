import time, keyboard
import importlib.util
import constants as cst
import numpy as np

from render_funcs import init_viz
from learning_utils import set_seed
from model import load_runner
from sensor.sensor import IMU_set
from multiprocessing import Process
set_seed(42)


''' Load Character Info Moudle '''
spec = importlib.util.spec_from_file_location("char_info", "amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)


init_grid_np = np.random.uniform(-10.0, 10.0, (cst.GRID_NUM, cst.GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

_, char, _, _, _, _ = init_viz(char_info,
                                          init_grid_list,
                                          hmap_scale=cst.GRID_SIZE,
                                          compare_gt=False)


if __name__ == '__main__':
    imu_set = IMU_set()
    sensor_process = Process(target=imu_set.processing)
    sensor_process.start()
    
    
    s_init_T_pose = np.zeros(cst.n_dofs * 2)
    s_init_T_pose[2] = 0.85
    s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])

    rt_runner = load_runner(
        s_init_T_pose,
        character=char,
        minimal=True,
        max_win_len=90
    )
    last_root_pos = s_init_T_pose[:3]
    imu_set.clear()
    
    flag_runing = True
    while flag_runing:
        while imu_set.empty():
            # if keyboard.is_pressed('q'):
            #     flag_runing = False
            #     break
            time.sleep(0.001)
        
        frame = imu_set.get()
        res = rt_runner.step(frame, last_root_pos)
        last_root_pos[:] = res['qdq'][:3]


    sensor_process.kill()
