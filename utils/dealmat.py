import h5py as h5
import torch
import numpy as np

def readmat(path): 
    p1 = 173
    nr1 = 150
    nc1 = 110
    with h5.File(path) as f:
        data = [f[element[0]][:] for element in f['Y']]
    data = np.array(data)
    HSI = data.transpose(0, 2, 1).reshape([1, 6, p1, nr1, nc1], order='F')
    HSI = torch.tensor(HSI).float()
    return HSI
