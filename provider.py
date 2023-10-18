import numpy as np
from fairmotion.ops import conversions
import torch

pi = np.pi
FREQ = 60

# Based from TransPose github repo
class IMUSet:
    def __init__(self):
        X = np.zeros(shape=72, dtype=np.float32)
        arr = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32).reshape(9)
        for i in range(6):
            st = i*9
            X[st:st+9] = arr
        
        self.X = X
        print("Define X:", X)
        self.create_data()
        # self.our_data()
        # self.wireless_data()
        # self.single_wireless()

        self.count = 0
        self.length = len(self.data)
        print("Data length:", self.length)
    
    def current_reading(self):
        self.count += 1
        return self.data[self.count-1]
    
    def available(self):
        return True if self.count < self.length else False

    def create_data(self):
        self.data = []
        # Create data Root and T-pose
        for i in range(FREQ*7):
            self.data.append(self.X.copy())
            
        k = 300
        X = self.X.copy()
        # Rotate X coordination, Hand down and up
        for i in range(k):
            X[18:27] = conversions.E2R([pi/3/k*i, 0, pi/3/k*i]).reshape(9)
            self.data.append(X.copy())

        for i in range(k):
            X[18:27] = conversions.E2R([pi/3/k*(k-i), 0, pi/3/k*(k-i)]).reshape(9)
            self.data.append(X.copy())
            
        # for i in range(k):
        #     X[18:27] = conversions.E2R([pi/3/k*i, 0, 0]).reshape(9)
        #     self.data.append(X.copy())

        # for i in range(k):
        #     X[18:27] = conversions.E2R([pi/3/k*(k-i), 0, 0]).reshape(9)
        #     self.data.append(X.copy())

        # Rotate Z coordination, Hand 
        for i in range(k):
            X[18:27] = conversions.E2R([0, 0, pi/2/k*i]).reshape(9)
            self.data.append(X.copy())
        for i in range(k):
            X[18:27] = conversions.E2R([0, 0, pi/2/k*(k-i)]).reshape(9)
            self.data.append(X.copy())
        
        print("Create data DONE")
    
    def our_data(self):
        self.data = []
        X = self.X.copy()

        data_path = "../Hi229/data/AnhPN/"
        for acc, quat in torch.load(data_path + "root.pt"):
            rotation = conversions.Q2R(quat).reshape(9)
            X[:54] = np.tile(rotation, 6)
            X[54:] = np.tile(acc/1000, 6)
            self.data.append(X.copy())
        
        for acc, quat in torch.load(data_path + "2023-07-29 15-31.pt"):
            X[18:27] = conversions.Q2R(quat).reshape(9)
            X[60:63] = acc/1000
            self.data.append(X.copy())
        
        print("Create data DONE")

    def wireless_data(self):
        self.data = []
        X = self.X.copy()

        data_path = "../Hi229/data/AnhPN/"
        # for acc, quat in torch.load(data_path + "root_wireless.pt"):
        for ind, acc, quat in torch.load(data_path + "2023-10-17 23-30.pt"):
            for i in ind:
                st = i*9
                X[st:st+9] = conversions.Q2R(quat[i]).reshape(9)
                X[0:9] = conversions.Q2R(quat[i]).reshape(9)

                st = i*3 + 54
                X[st:st+3] = acc[i]
            self.data.append(X.copy())

            # rotation = conversions.Q2R(quat).reshape(9)
            # X[:54] = np.tile(rotation, 6)
            # X[54:] = np.tile(acc, 6)
            # self.data.append(X.copy())
        
        for ind, acc, quat in torch.load(data_path + "2023-07-29 15-27.pt"):
            for i in ind:
                st = i*9
                X[st:st+9] = conversions.Q2R(quat[i]).reshape(9)

                st = i*3 + 54
                X[st:st+3] = acc[i]
            self.data.append(X.copy())
        
        print("Create data DONE")

    def single_wireless(self):
        self.data = []
        X = self.X.copy()
# ([2], [array([-0.036,  0.045,  1.018], dtype=float32)], [array([ 0.257, -0.008,  0.028,  0.966], dtype=float32)])
        data_path = "../Hi229/data/AnhPN/"
        # for acc, quat in torch.load(data_path + "root_wireless.pt"):
        for ind, acc, quat in torch.load(data_path + "2023-10-17 23-41.pt"):
            i = ind[0]
            st = i*9
            X[st:st+9] = conversions.Q2R(quat[0]).reshape(9)
            X[0:9] = conversions.Q2R(quat[0]).reshape(9)

            st = i*3 + 54
            X[st:st+3] = acc[0]
            X[54:57] = acc[0]
            
            self.data.append(X.copy())

            # rotation = conversions.Q2R(quat).reshape(9)
            # X[:54] = np.tile(rotation, 6)
            # X[54:] = np.tile(acc, 6)
            # self.data.append(X.copy())
        
        for ind, acc, quat in torch.load(data_path + "2023-10-17 23-42.pt"):
            i = ind[0]
            st = i*9
            X[st:st+9] = conversions.Q2R(quat[0]).reshape(9)

            st = i*3 + 54
            X[st:st+3] = acc[0]
            
            self.data.append(X.copy())
        
        print("Create data DONE")
