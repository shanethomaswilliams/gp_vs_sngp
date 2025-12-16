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
def load_data(file_path, sample_size, train_percentage = 0.70, random_state=42, shuffle = True,
              num_test = 50_000):
    #Makes sure the splits make sense
    assert train_percentage < 1.0


    #Load data
    tensor_DF = torch.load(file_path)
    x_DF = tensor_DF["x"].float()
    y_D = tensor_DF["y"].float()

    
    # Random shuffle of row ids corresponding to all L provided examples
    D = len(x_DF)
    rng = np.random.default_rng(seed=random_state)
    shuffled_ids_D = rng.permutation(np.arange(D))

    #Train = N, Validation = V, Test = T
    #Determine N, V, T
    N = int(np.round(sample_size * (train_percentage)))
    V = sample_size - N
    assert D >= N + V

    #Make x_data
    if x_DF.dim() == 1:
        x_train_NF = x_DF[shuffled_ids_D[0:N]].unsqueeze(-1)
        x_valid_VF = x_DF[shuffled_ids_D[N:N+V]].unsqueeze(-1)
        x_GP_train_N = x_DF[shuffled_ids_D[0:N+V]].unsqueeze(-1)
        x_test_TF = x_DF[shuffled_ids_D[-num_test:]].unsqueeze(-1)
    else:
        x_train_NF = x_DF[shuffled_ids_D[0:N]]
        x_valid_VF = x_DF[shuffled_ids_D[N:N+V]]
        x_GP_train_N = x_DF[shuffled_ids_D[0:N+V]]
        x_test_TF = x_DF[shuffled_ids_D[-num_test:]]


    #Make SNGP y_data
    y_SNGP_train_N1 = y_D[shuffled_ids_D[0:N]].unsqueeze(-1)
    y_SNGP_valid_V1 = y_D[shuffled_ids_D[N:N+V]].unsqueeze(-1)
    y_SNGP_test_T1 = y_D[shuffled_ids_D[-num_test:]].unsqueeze(-1)

    #Make GP y_data
    y_GP_train_N = y_D[shuffled_ids_D[0:N+V]].squeeze()
    y_GP_test_T = y_D[shuffled_ids_D[-num_test:]].squeeze()

    #Creates datasets
    GP_train_dataset = TensorDataset(x_GP_train_N, y_GP_train_N)
    GP_test_dataset = TensorDataset(x_test_TF, y_GP_test_T)

    SNGP_train_dataset = TensorDataset(x_train_NF, y_SNGP_train_N1)
    SNGP_valid_dataset = TensorDataset(x_valid_VF, y_SNGP_valid_V1)
    SNGP_test_dataset = TensorDataset(x_test_TF, y_SNGP_test_T1)

    GP_train_loader = DataLoader(GP_train_dataset, batch_size=N+V, shuffle=shuffle)
    GP_test_loader = DataLoader(GP_test_dataset, batch_size=num_test, shuffle=shuffle)

    SNGP_train_loader = DataLoader(SNGP_train_dataset, batch_size=N, shuffle=shuffle)
    SNGP_valid_loader = DataLoader(SNGP_valid_dataset, batch_size=V, shuffle=shuffle)
    SNGP_test_loader = DataLoader(SNGP_test_dataset, batch_size=num_test, shuffle=shuffle)

    return GP_train_loader, GP_test_loader, SNGP_train_loader, SNGP_valid_loader, SNGP_test_loader


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
