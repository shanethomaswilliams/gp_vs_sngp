import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

'''
Inputs:
    file_path: filename to load data from
    sample_size: number of samples to receive
    train_percentage: What percent of data should be in training loader
                      the rest will be allocated
    random_state: Seed used for reproducable randomness
    Shuffle: to Shuffle the data loaders or not

Returns: 
    GP_train_loader: Train loader formated for GP x is (N, F) 
    GP_valid_loader: Val loader formated for GP y is (N, 1)
    SNGP_train_loader: Train loader formated for SNGP x is (N, F)
    SNGP_valid_loader: Val Loader formated for SNGP y is (N)
    
'''
def load_data_train(file_path, sample_size, train_percentage = 0.70, random_state=42, shuffle = True,
              num_test = 10_000, standardize_x=False, standardize_y=False, eps=1e-8):

    tensor_DF = torch.load(file_path)
    x_DF = tensor_DF["x"].float()
    y_D  = tensor_DF["y"].float()

    D = len(x_DF)
    rng = np.random.default_rng(seed=random_state)
    ids = rng.permutation(np.arange(D))

    N = int(np.round(sample_size * train_percentage))
    V = sample_size - N

    train_ids = ids[:N]
    val_ids   = ids[N:N+V]
    test_ids  = ids[-num_test:]

    # ---- slice raw ----
    def slice_x(ix):
        x = x_DF[ix]
        return x.unsqueeze(-1) if x.dim() == 1 else x

    x_train = slice_x(train_ids)
    x_val   = slice_x(val_ids)
    x_test  = slice_x(test_ids)

    y_train = y_D[train_ids]
    y_val   = y_D[val_ids]
    y_test  = y_D[test_ids]

    # ---- TRAIN-ONLY normalization ----
    if standardize_x:
        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std  = x_train.std(dim=0, keepdim=True).clamp_min(eps)
        x_train = (x_train - x_mean) / x_std
        x_val   = (x_val   - x_mean) / x_std
        x_test  = (x_test  - x_mean) / x_std

    if standardize_y:
        y_mean = y_train.mean()
        y_std  = y_train.std().clamp_min(eps)
        y_train = (y_train - y_mean) / y_std
        y_val   = (y_val   - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std

    # ---- build loaders ----
    GP_train_dataset = TensorDataset(torch.cat([x_train, x_val], dim=0),
                                     torch.cat([y_train, y_val], dim=0))
    GP_test_dataset  = TensorDataset(x_test, y_test)

    SNGP_train_dataset = TensorDataset(x_train, y_train.unsqueeze(-1))
    SNGP_valid_dataset = TensorDataset(x_val,   y_val.unsqueeze(-1))
    SNGP_test_dataset  = TensorDataset(x_test,  y_test.unsqueeze(-1))

    GP_train_loader = DataLoader(GP_train_dataset, batch_size=len(GP_train_dataset), shuffle=False)
    GP_test_loader  = DataLoader(GP_test_dataset,  batch_size=len(GP_test_dataset),  shuffle=False)

    SNGP_train_loader = DataLoader(SNGP_train_dataset, batch_size=len(SNGP_train_dataset), shuffle=shuffle)
    SNGP_valid_loader = DataLoader(SNGP_valid_dataset, batch_size=len(SNGP_valid_dataset), shuffle=False)
    SNGP_test_loader  = DataLoader(SNGP_test_dataset,  batch_size=len(SNGP_test_dataset),  shuffle=False)

    if standardize_x and standardize_y:
        norm_stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    else:
        norm_stats = None
    return GP_train_loader, GP_test_loader, SNGP_train_loader, SNGP_valid_loader, SNGP_test_loader, norm_stats



'''

WE SHOULD USE THIS FUNCTION ANYMORE

Inputs:
    file_path: filename to load data from
    sample_size: number of samples to receive
    random_state: Seed used for reproducable randomness
    Shuffle: to Shuffle the data loaders or not

Returns: 
    GP_test_loader: Train loader formated for GP
    SNGP_test_loader: Train loader formated for SNGP 
'''
def load_data_test(file_path, sample_size, random_state=42, shuffle = True, exclude_list = torch.empty):

    #Load data
    tensor_DF = torch.load(file_path)
    x_DF = tensor_DF["x"].float()
    y_D = tensor_DF["y"].float()

    D = len(x_DF)
    exMask = torch.full(size = D, fill_value=True)

    for d, x in enumerate(x_DF):
        exMask[d] = ()

    exMask = torch.isin(x_DF, exclude_list)
    avaliable_Xvals = x_DF[exMask]
    avaliable_Yvals = y_D[exMask]

    # Random shuffle of row ids corresponding to all L provided examples
    A = len(avaliable_Xvals)
    N = sample_size
    rng = np.random.default_rng(seed=random_state)
    shuffled_ids_A = rng.permutation(np.arange(A))

    #Train = N, Validation = V, Test = T
    #Determine N, V, T
    assert A >= N

    #Make x_data
    if x_DF.dim() == 1:
        x_test_NF = avaliable_Xvals[shuffled_ids_A[0:N]].unsqueeze(-1)
    else:
        x_test_NF = avaliable_Xvals[shuffled_ids_A[0:N]]

    #Make SNGP y_data
    y_SNGP_test_N1 = avaliable_Yvals[shuffled_ids_A[0:N]].unsqueeze(-1)

    #Make GP y_data
    y_GP_test_N = y_SNGP_test_N1.squeeze()

    #Creates datasets
    GP_test_dataset = TensorDataset(x_test_NF, y_GP_test_N)
    SNGP_test_dataset = TensorDataset(x_test_NF, y_SNGP_test_N1)

    GP_test_loader = DataLoader(GP_test_dataset, batch_size=N, shuffle=shuffle)

    SNGP_test_loader = DataLoader(SNGP_test_dataset, batch_size=N, shuffle=shuffle)

    return GP_test_loader, SNGP_test_loader
