import torch as th
from torch.utils import data
import pickle as pkl
import numpy as np

class Loader(data.Dataset):
    def __init__(self, split="train", transform=None, samples=10):
        self.split   = split
        self.samples = samples
        if self.split == "train":
            with open("./data/train.pkl", "rb") as f:
                self.data = pkl.load(f)

                self.data = [self.data[i]["lidar"]["measure"] for i in range(len(self.data))]
                self.data = np.array(self.data, dtype=float)
                #print(self.data.shape)
                #print(self.data[2,0,:]/ 1000)
                self.data = np.reshape(self.data, (-1, self.data.shape[2]))
                self.data /= 1000
                self.data[self.data > 2.0] = 2.0
                #print(self.data[200,:])
                #self.data = [self.data[(index*self.samples):((index+1)*self.samples)] for index in range(int(self.data.shape[0]/self.samples))]
        else:
            with open("./data/test.pkl", "rb") as f:
                self.data = pkl.load(f)[0]["data"]
                self.data = np.array([float(i/1000) for i in self.data])
                #self.data = [self.data[(index*self.samples):((index+1)*self.samples)] for index in range(int(len(self.data)/self.samples))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index < 10:
            index += 10
        lidar = self.data[(index - self.samples):index]
        #print(lidar)
        if self.split == "train":
            lidar = lidar.flatten()
            #print(lidar)
            lidar = th.from_numpy(lidar)
            lidar = lidar.float()
        else:
            # everything background
            n = lidar.shape[0]
            lidar_array = np.ones(lidar.shape[0]*9) * 2.0
            # lidar 3
            k = 3
            lidar_array[(k*n):(k+1)*n] = lidar
            #lidar_array[:, k] = lidar
            #print(lidar_array.shape)
            #lidar_array = np.array(lidar_array)
            lidar_array = th.from_numpy(lidar_array)
            lidar = lidar_array.float()
        return lidar

if __name__ == '__main__':

    data = Loader("train")
    #print(len(data))
    #print(data[100])
    #data = Loader("tr")
    #print(data[100])