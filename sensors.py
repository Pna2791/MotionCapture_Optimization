import time
import numpy as np
import json
import serial
from multiprocessing import Queue
from fairmotion.ops import conversions

from hipnuc_protocol import (
    intercept_one_complete_frame,
    extraction_information_from_frame,
    find_frameheader,
    HipnucFrame_NotCompleted_Exception,
    HipnucFrame_ErrorFrame_Exception
)



MAX_ACC = 10.0


class IMU(object): 
 
    def __init__(self, portx='COM6', bps=460800, path_configjson='config_wireless.json') -> None: 
        config_json = open(path_configjson, 'r', encoding='utf-8')
        self.config = json.load(config_json)
        config_json.close()
        self.portx = portx
        self.bps = bps
    
        self.data_queue = Queue()
        self.binbuffer = []
        self.running = True
     
    def read(self): 
        data = self.data_queue.get() 
        return data
     
    
    def clear(self):
        while self.data_queue.qsize():
            self.data_queue.get()
        print("Queue length:", self.data_queue.qsize())
    
    def preprocess(self):
        
        def get_mean(path):
            data = np.load(path)
            
            mean_buffer = []
            for frame in data:
                acc = frame['acc']
                quat = frame['quat']
                mean_buffer.append(
                    np.concatenate((
                        conversions.Q2R(quat).reshape(-1),
                        acc.reshape(-1)
                    ))
                )
            return np.array(mean_buffer).mean(axis=0)
        
        R_Gp_B0 = np.array([
            1.0, 0, 0, 0, 1, 0, 0, 0, 1,
            1.0, 0, 0, 0, 1, 0, 0, 0, 1,
            1.0, 0, 0, 0, 1, 0, 0, 0, 1,
            1.0, 0, 0, 0, 1, 0, 0, 0, 1,
            1.0, 0, 0, 0, 1, 0, 0, 0, 1,
            1.0, 0, 0, 0, 1, 0, 0, 0, 1,
        ])
        
        R_and_acc_mean = get_mean('data/root.npy')
        R_Gn_Gp = R_and_acc_mean[:6*9].reshape((6, 3, 3))
        self.acc_offset_Gp = R_and_acc_mean[6*9:].reshape(6, 3) 
        
        R_and_acc_mean = get_mean('data/t_pose.npy')
        R_Gn_S0 = R_and_acc_mean[: 6 * 9].reshape((6, 3, 3))
        
        
        R_Gp_S0 = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_S0)
        R_B0_S0 = np.einsum('nij,njk->nik', R_Gp_B0.transpose((0, 2, 1)), R_Gp_S0)
        
        self.R_Gn_Gp = R_Gn_Gp.transpose((0, 2, 1))
        self.R_B0_S0 = R_B0_S0.transpose((0, 2, 1))
        
        # sample = {
        #     'GWD': [{'': 0}], 'id': [{'': 0}, {'': 1}, {'': 2}, {'': 3}, {'': 4}, {'': 5}],
        #     'timestamp': [{'(s)': 0.0}, {'(s)': 0.0}, {'(s)': 0.0}, {'(s)': 0.0}, {'(s)': 0.0}, {'(s)': 0.0}],
        #     'acc': [
        #         {'X': 0.0, 'Y': 0.024, 'Z': 1.012}
        #         {'X': 0.018, 'Y': 0.045, 'Z': 1.02},
        #         {'X': 0.014, 'Y': 0.06, 'Z': 1.017},
        #         {'X': 0.016, 'Y': -0.008, 'Z': 1.011},
        #         {'X': 0.017, 'Y': 0.019, 'Z': 1.008},
        #         {'X': 0.038, 'Y': 0.015, 'Z': 0.992}
        #     ],
        #     'gyr': [{'X': 0.0, 'Y': 0.2, 'Z': 0.0}, {'X': 0.0, 'Y': 0.0, 'Z': 0.0}, {'X': 0.5, 'Y': 0.0, 'Z': -0.1}, {'X': -0.5, 'Y': 0.3, 'Z': 0.4}, {'X': -0.2, 'Y': -0.1, 'Z': 0.0}, {'X': 0.0, 'Y': 0.1, 'Z': 0.0}],
        #     'mag': [{'X': -38, 'Y': 13, 'Z': -1}, {'X': -23, 'Y': 10, 'Z': 2}, {'X': -28, 'Y': -1, 'Z': -3}, {'X': -33, 'Y': -2, 'Z': 7}, {'X': -30, 'Y': -3, 'Z': 0}, {'X': -25, 'Y': 12, 'Z': -1}],
        #     'euler': [{'Roll': 1.27, 'Pitch': 0.11, 'Yaw': -64.98}, {'Roll': 2.44, 'Pitch': -1.04, 'Yaw': -41.68}, {'Roll': 3.12, 'Pitch': -0.71, 'Yaw': -48.42}, {'Roll': -1.1, 'Pitch': -0.37, 'Yaw': -18.62}, {'Roll': 1.18, 'Pitch': -0.77, 'Yaw': -96.09}, {'Roll': 1.3, 'Pitch': -2.03, 'Yaw': -48.77}],
        #     'quat': [
        #         {'W': 0.843, 'X': 0.01, 'Y': -0.005, 'Z': -0.537},
        #         {'W': 0.934, 'X': 0.017, 'Y': -0.016, 'Z': -0.355},
        #         {'W': 0.912, 'X': 0.022, 'Y': -0.017, 'Z': -0.41},
        #         {'W': 0.987, 'X': -0.01, 'Y': -0.002, 'Z': -0.162},
        #         {'W': 0.669, 'X': 0.002, 'Y': -0.012, 'Z': -0.744},
        #         {'W': 0.911, 'X': 0.003, 'Y': -0.021, 'Z': -0.413}
        #     ]
        # }
    
    def process(self, data):
        acc = data['acc']
        acc = np.array(
            [[X['X'], X['Y'], X['Z']] for X in acc],
            dtype=np.float32
        )
        
        quat = data['quat']
        quat = np.array(
            [[X['X'], X['Y'], X['Z'], X['W']] for X in quat],
            dtype=np.float32 
        )
        
        if self.running:
            R_Gn_St = conversions.Q2R(quat)
            
            R_Gp_St = np.einsum('nij,njk->nik', self.R_Gn_Gp, R_Gn_St)
            R_Gp_Bt = np.einsum('nij,njk->nik', R_Gp_St, self.R_B0_S0)

            acc_Gp = np.einsum('ijk,ik->ij', R_Gp_St, acc)
            acc_Gp = acc_Gp - self.acc_offset_Gp

            acc_Gp = np.clip(acc_Gp, -MAX_ACC, MAX_ACC)
            
            self.data_queue.append(
                np.concatenate((R_Gp_Bt.reshape(-1), acc_Gp.reshape(-1)))
            )
        else:
            self.data_queue.append({
                'acc': acc,
                'quat': quat
            })
        
    def running(self):
        print("RUNING", "==="*30)
        self.serial = serial.Serial(self.portx, self.bps, timeout=None)
        while  True: 
            if self.serial.in_waiting:
                data = self.serial.read(self.serial.in_waiting)
                self.binbuffer.extend(data)

            try:
                while True:
                    # print("In loop", "==="*30)
                    headerpos, endpos = intercept_one_complete_frame(self.binbuffer)
                    data = extraction_information_from_frame(
                        self.binbuffer[headerpos :endpos + 1],
                        self.config["report_datatype"]
                    )
                    self.process(data)
                    self.binbuffer = self.binbuffer[endpos + 1:]

            except HipnucFrame_NotCompleted_Exception as NotCompleted:
                pass
            except HipnucFrame_ErrorFrame_Exception as e:
                headerpos = find_frameheader(self.binbuffer)
                self.binbuffer = self.binbuffer[headerpos + 1:]
                
                
            time.sleep(0.001)
            