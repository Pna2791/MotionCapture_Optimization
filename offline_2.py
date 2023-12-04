import pickle, time
import importlib.util
import constants as cst
import numpy as np

from render_funcs import init_viz
from learning_utils import set_seed
from model import load_runner, Rendering
from data_queue_provider import Data_provider
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
    # test_file = 'data/preprocessed_DIP_IMU_v1/dipimu_s_03_01.pkl'
    # data = pickle.load(open(test_file, "rb"))

    # frames = 3000
    # X = data['imu'][:frames]
    data_provider = Data_provider()
    sub_processor = Process(target=data_provider.processing)
    sub_processor.start()
    
    
    s_init_T_pose = np.zeros(cst.n_dofs * 2)
    s_init_T_pose[2] = 0.85
    s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])

    rt_runner = load_runner(
        s_init_T_pose,
        character=char,
        minimal=True
    )

    last_root_pos = s_init_T_pose[:3]
    # n_length = len(X)
    n_length = 3000
    t_start = time.time()   
    # for t, frame in enumerate(X):
    while data_provider.available():
        frame = data_provider.get()
        res = rt_runner.step(frame, last_root_pos)
        last_root_pos[:] = res['qdq'][:3]

    print('Duration:', time.time() - t_start)
    print('FPS:', n_length/(time.time()-t_start))
    sub_processor.kill()