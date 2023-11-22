import multiprocessing 
import time 
import numpy as np 
from sensors import IMU


if __name__ == '__main__': 
    imu_set = IMU()
    
    sensor_process = multiprocessing.Process(target=imu_set.running) 
    sensor_process.start() 
 
    print("Starting") 
    try: 
        while True: 
            user_command = input("Enter command (exit): ") 
            if user_command == "exit": 
                break 
            elif user_command == "read": 
                t_start = time.time()
                for i in range(100):
                    data = imu_set.read() 
                    
                print("Data:", data)
                print(time.time() - t_start, imu_set.data_queue.qsize())
            elif user_command == "calib":
                imu_set.clear()
            else: 
                print("Invalid command. Try again.")
            
    except Exception as e: 
        print("ERROR:", e) 
    
    sensor_process.kill()
    print("Exited")