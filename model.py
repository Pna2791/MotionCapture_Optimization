
from real_time_runner_minimal import RTRunnerMin
from real_time_runner import RTRunner
import constants as cst


from bullet_agent import SimAgent

import onnxruntime as ort


# Hyper parameters
WITH_ACC_SUM = True
USE_5_SBP = True


def load_onnx_model(onnx_path:str):
    providers = [
        ("CUDAExecutionProvider", {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2*(1024**3),
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True
        })
        ]
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    return ort_session


def load_runner(
    s_init_T_pose,
    character: SimAgent,
    max_win_len:int = 60,
    model_path:str = "output/model-with-dip9and10-cpu-dynamic.onnx",
    minimal:bool = True,
):
    model = load_onnx_model(model_path)
    
    if minimal:
        return RTRunnerMin(
            character, model, max_win_len, s_init_T_pose,
            with_acc_sum=WITH_ACC_SUM,
        )
    else:
        return RTRunner(
            character, model, max_win_len, s_init_T_pose,
            map_bound=cst.MAP_BOUND,
            grid_size=cst.GRID_SIZE,
            play_back_gt=False,
            five_sbp=USE_5_SBP,
            with_acc_sum=WITH_ACC_SUM,
            multi_sbp_terrain_and_correction=False
        )


class Rendering:
    
    def __init__(self, pb_client, VIDs) -> None:
        self.pb_client = pb_client
        self.VIDs = VIDs
    
    def show(self, x, ind):
        self.pb_client.resetBasePositionAndOrientation(
            self.VIDs[ind],
            x,
            [0., 0, 0, 1]
        )