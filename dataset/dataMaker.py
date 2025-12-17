import numpy as np
import math
import torch
from sklearn.datasets import make_friedman3
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def makeSinPT(num_examples, random_state = None,
               noise = 0.1, num_gaps = 0, shuffle = True, dom_range = (-3,3),
               filename = None):
    
    input_length = dom_range[1] - dom_range[0]

    #Make random number generator with seed
    if random_state==None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=random_state)

    #determine where the gaps in the data will be
    gap_locs = np.zeros((num_gaps,2))
    for i in range(num_gaps):
        lower_bound = np.round(rng.uniform(),2)
        upper_bound = np.round(min(1.0, lower_bound + rng.uniform(low=0.05, high=0.10), 2),2)

        lower_bound = dom_range[0] + (lower_bound * input_length)
        upper_bound = dom_range[0] + (upper_bound * input_length)
        gap_locs[i] = (lower_bound, upper_bound)

    candidates = rng.uniform(dom_range[0], dom_range[1], num_examples)

    # Build mask for excluded ranges
    mask = np.ones_like(candidates, dtype=bool)
    for start, end in gap_locs:
        mask &= ~((candidates >= start) & (candidates <= end))

    valid = candidates[mask]

    # If not enough valid points, resample until we have enough
    while valid.size < num_examples:
        extra = rng.uniform(dom_range[0], dom_range[1], num_examples)
        extra_mask = np.ones_like(extra, dtype=bool)
        for start, end in gap_locs:
            extra_mask &= ~((extra >= start) & (extra <= end))
        valid = np.concatenate([valid, extra[extra_mask]])

    #get data
    x = valid[:num_examples]
    y = np.sin(2 * x) - np.cos(x) + noise * rng.normal(size = num_examples)


    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    data_dict = {'x':X_tensor,
                 'y':y_tensor}
    
    if filename != None:
        torch.save(data_dict, filename)

    return data_dict


def makeCrazySinPT(num_examples, random_state = None,
                   noise = 0.1, shuffle = True, dom_range = (-3,3),
                   filename = None):
    #Make random number generator with seed
    if random_state==None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=random_state) 

    x_vals = rng.uniform(low = dom_range[0], high=dom_range[1], size = num_examples)
    y_vals = np.sin(10 * x_vals) - np.cos(7 * x_vals) + noise * rng.normal(size = num_examples)

    X_tensor = torch.tensor(x_vals, dtype=torch.float32)
    y_tensor = torch.tensor(y_vals, dtype=torch.float32)

    data_dict = {'x':X_tensor,
                 'y':y_tensor}
    
    if filename != None:
        torch.save(data_dict, filename)

    return data_dict

def makeFriedmanPT(n_samples, random_state=None,
                         noise = 0.1, shuffle = True,
                         filename = None):
    
    if random_state==None:
        X, y = make_friedman3(n_samples=n_samples, noise=noise)
    else:
        X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    data_dict = {'x':X_tensor,
                 'y':y_tensor}
    
    if filename != None:
        torch.save(data_dict, filename)

    return data_dict

def makeFriedman1(n_samples, random_state=None,
                         noise = 0.1, shuffle = True,
                         filename = None):
    
    if random_state==None:
        X, y = make_friedman1(n_samples=n_samples, n_features=5, noise=noise)
    else:
        X, y = make_friedman1(n_samples=n_samples, n_features=5, noise=noise, random_state=random_state)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    data_dict = {'x':X_tensor,
                 'y':y_tensor}
    
    if filename != None:
        torch.save(data_dict, filename)

    return data_dict
