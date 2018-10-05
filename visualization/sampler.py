import pickle as pkl 
import numpy as np


class Sampler:
    def __init__(self):
        with open("../data/train_data_correct.pkl", "rb") as f:
            self.data = pkl.load(f)

            self.runs = len(self.data)
            self.data_lidar = []
            for i in range(self.runs):       
                self.data_lidar.extend(self.data[i]["lidar"]["measurements"])

            self.data_lidar = np.array(self.data_lidar, dtype=float)
            self.data_lidar /= 1000
            self.data_lidar[self.data_lidar > 2.0] = 2.0

        self.index=0

    def get_sample(self):
        if self.index == self.data_lidar.shape[0]:
            self.index = -1
        self.index += 1
        return self.data_lidar[self.index]

    def get_dataset(self):
        return self.data_lidar


if __name__ == '__main__':
    s = Sampler()
    print(s.get_sample())
    print(s.get_dataset().shape)