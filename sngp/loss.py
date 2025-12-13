import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import math
import tqdm # type: ignore
from sklearn.datasets import make_moons # type: ignore
from sklearn import metrics # type: ignore
import torch # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import numpy as np # type: ignore

def square_loss(means, y):
    loss = nn.MSELoss(reduction='sum')
    return loss(means, y)