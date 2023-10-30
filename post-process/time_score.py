import numpy as np
from train import CNNClassifier
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import os
import re
import glob

def classification(index):
    phase = ['SG', 'DG', 'SD', 'DD', 'SP', 'DP', 'HEX','DIS']
    if index > len(phase) - 1:
        return None
    else:
        return phase[index]


def average_every_n_lists(y, n=10):
    # Convert the list of lists to a NumPy array
    arr = np.array(y)

    # Ensure the array has at least two dimensions
    if len(arr.shape) == 1:
        arr = arr[:, np.newaxis]

    # Calculate the number of chunks
    num_chunks = len(arr) // n

    # Reshape the data to a (-1, n) shape and then take the average along axis 1
    averaged_data = np.mean(arr[:num_chunks * n].reshape(-1, n, arr.shape[1]), axis=1).tolist()

    return averaged_data

def average_rolling_window(y, n=25):
    arr = np.array(y)

    # Ensure the array has at least two dimensions
    if len(arr.shape) == 1:
        arr = arr[:, np.newaxis]

    # Create a window of size n
    window = np.ones((n,)) / n

    # Apply the rolling window average using np.convolve and 'valid' mode
    averaged_data = [np.convolve(arr[:, i], window, 'valid') for i in range(arr.shape[1])]

    # Convert back to a list of lists (or a list of numbers if there's only one column)
    averaged_data = np.column_stack(averaged_data).tolist()

    return averaged_data


# Custom sort key function
def custom_sort_key(filename):
    # Use regex to extract numbers
    x, y = map(int, re.findall(r"(\d+)", filename))
    return (x, y)

def plot_p_t(idx,p_list,color_list,path):
    p_list=np.array(p_list)
    print(p_list.shape)
    # t_list=np.arange(0,1000,2)
    t_list = np.arange(0, p_list.shape[0] * 2, 2)

    #set font
    plt.rcParams['font.family'] = 'Arial'

    # Set global default text sizes
    plt.rcParams['font.size'] = 16  # Default text size
    plt.rcParams['axes.labelsize'] = 14  # Axes label size
    plt.rcParams['xtick.labelsize'] = 14  # X-tick label size
    plt.rcParams['ytick.labelsize'] = 14  # Y-tick label size
    plt.rcParams['legend.fontsize'] = 14  # Legend font size
    plt.rcParams['figure.titlesize'] = 16  # Title size for the whole figure
    plt.rcParams['axes.titlesize'] = 14  # Title size for individual axes

    plt.figure(figsize=(12,6))
    for i in range(len(p_list[0])):
        plt.plot(t_list,p_list[:,i],label=f'class {classification(i)}',color=color_list[i])
    plt.xlabel('Time (ns)')
    plt.ylabel('Softmax Probability')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    folder=path
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}{idx}.png',dpi=300,bbox_inches='tight')
    return None

#load data from dataset
def data_loader(file_path, dataset_idx):
    all_files = os.listdir(file_path)
    all_files = [ file for file in all_files if file.endswith('.npy') and file.startswith(f'{dataset_idx}_')]
    all_files=sorted(all_files,key=custom_sort_key)
    p_list = []
    t_list = []
    for file in all_files:
        time = int(file.split('_')[1].split('.')[0]) * 2
        t_list.append(time)
        temp = torch.from_numpy(np.load(os.path.join(file_path,file)))
        temp = temp.unsqueeze(0).unsqueeze(0).to(torch.float32)
        temp = model.forward(temp).squeeze(0)
        p = F.softmax(temp, dim=0).tolist()
        p_list.append(p)
    print(p_list)
    return p_list


def average_scores(model_paths, data_path, dataset_idx):
    # average scores over 10 folds
    p_list_sum = None
    for model_path in model_paths:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        # load the weight into the model
        model.load_state_dict(new_state_dict)
        model.eval()

        p_list = data_loader(data_path, dataset_idx)

        if p_list_sum is None:
            p_list_sum = np.array(p_list)
        else:
            p_list_sum += np.array(p_list)

    # Average the scores
    p_list_avg = p_list_sum / len(model_paths)
    return p_list_avg.tolist()

if __name__ == "__main__":
    model = CNNClassifier(8)
    # Load the model weights from the .pt file
    model_folder = "../../Documents/CNN-for-MD/trained_models/model_soft_0828/"
    model_paths = [os.path.join(model_folder, f"fold_{i+1}_epoch_140.pt") for i in range(10)]
    model_paths = [os.path.join(model_folder, f"fold_{i + 1}_epoch_140.pt") for i in [1,8,9]]

    #load data
    data_path = '../../Documents/CNN-for-MD/test_data/data_0920/'
    out_path = os.path.join(os.path.dirname(model_folder),'average_1_9_10/')
    dataset=np.arange(84,88)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    with torch.no_grad():
        for idx in dataset:
            # p_list,t_list=data_loader(data_path,idx)
            # _ = plot_p_t(idx, p_list, t_list, color_list, out_path)
            p_list = average_scores(model_paths, data_path, idx)
            _ = plot_p_t(idx, p_list, color_list, out_path)






