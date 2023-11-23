from real_time_runner_minimal import RTRunnerMin
from real_time_runner import RTRunner
import constants as cst

from bullet_agent import SimAgent

from scipy.spatial.transform import Rotation
import onnxruntime as ort
import tkinter as tk
import numpy as np


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


class Monitor:
    
    def __init__(self) -> None:
        self.XX = [100, 0, 200, 0, 200, 100]
        self.YY = [300, 150, 150, 450, 450, 0]
            
            
        self.window = tk.Tk()
        self.window.title("Frame Display")

        # Create a canvas to display your data
        self.canvas = tk.Canvas(self.window, width=400, height=600)
        self.canvas.pack()
        self.t = 0
    
    def show(self, frame):
        font = 5
        self.canvas = None
        rot = frame[:54].reshape((6, 3, 3))
        acc = frame[54:].reshape((6, 3))
        # Display each matrix in the frame
        self.canvas.delete("all")
        self.canvas.create_text(
            50, 10,
            text=str(self.t),
            font=font
        )
        for i in range(6):
            r = Rotation.from_matrix(rot[i])
            euler_angles = np.degrees(r.as_euler('xyz'))

            for col in range(3):
                for row in range(3):
                    value = rot[i][row][col]
                    self.canvas.create_text(
                        self.XX[i] + col * 40 +50,
                        self.YY[i] + row * 20 +30,
                        text="{:.2f}".format(value),
                        font=font
                    )
                self.canvas.create_text(
                    self.XX[i] + col * 40 +50,
                    self.YY[i] + 3 * 20 +30,
                    text="{}".format(round(euler_angles[col])),
                    font=font
                )
                self.canvas.create_text(
                    self.XX[i] + col * 40 +50,
                    self.YY[i] + 4 * 20 +30,
                    text="{:.2f}".format(acc[i, col]),
                    font=font
                )
            
        self.window.update()
        self.t += 1
