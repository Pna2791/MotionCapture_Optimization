from multiprocessing import Process
import time, sys, pickle
import numpy as np 
from sensors import IMU


def show(mess):
    sys.stdout.write(f"\r{mess}")
    sys.stdout.flush()

if __name__ == '__main__': 
    imu_set = IMU()
    imu_set.is_running = False
    
    sensor_process = Process(target=imu_set.processing) 
    sensor_process.start()
    
    root = True
    root = False
    
    file_name = 'data/root.pkl' if root else 'data/t_pose.pkl'
    
    for i in range(-3, 0):
        show(f"Start in {i} seconds")
        time.sleep(1)
    
    imu_set.clear()
    print("Starting", "==="*30) 
    try:
        for i in range(3):
            show(f"Record at {i+1} second")
            time.sleep(1)
            
        pack = list()
        for i in range(180):
            data = imu_set.get()
            pack.append(data)
        print(data)

        with open(file_name, 'wb') as f:
            pickle.dump(pack, f)
        print("SAVED at:", file_name)
            
    except Exception as e: 
        print("ERROR:", e) 
    
    sensor_process.kill()
    print("Exited")