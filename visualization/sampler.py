import pickle as pkl 
import numpy as np


class Sampler:
    def __init__(self, n=100):
        with open("../data/anomaly_detection_gt.pkl", "rb") as f:
            self.data = pkl.load(f)

            self.runs = len(self.data)
            self.data_lidar   = []
            self.data_joint   = []
            self.data_joint_v = []
            for i in range(self.runs):       
                self.data_lidar.extend(self.data[i]["lidar"]["measurements"])
                self.data_joint.extend(self.data[i]["state"]["j_pos"])
                self.data_joint_v.extend(self.data[i]["state"]["j_vel"])
            
            self.data_lidar = np.array(self.data_lidar, dtype=float)
            self.data_lidar /= 1000
            self.data_lidar[self.data_lidar > 2.0] = 2.0

            self.data_joint = np.array(self.data_joint, dtype=float)
            self.data_joint_v = np.array(self.data_joint_v, dtype=float)

        self.index=0
        self.n = 100

    def get_sample(self):
        if self.index == (self.data_lidar.shape[0] - self.n):
            self.index = 0
        sample_lidar   = self.data_lidar[(self.index):(self.index+self.n)]
        sample_joint   = self.data_joint[(self.index):(self.index+self.n)]
        sample_joint_v = self.data_joint_v[(self.index):(self.index+self.n)]
        self.index += self.n
        return (sample_lidar, sample_joint, sample_joint_v)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    s = Sampler()
    for i in range(10):
        dataset, _, _ = s.get_sample()
        plt.plot(dataset[:,3])
        plt.ylim([0,2])
        plt.show()