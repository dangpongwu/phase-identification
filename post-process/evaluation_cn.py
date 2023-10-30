import numpy as np
from train.train import CNNClassifier
import torch
import torch.nn.functional as F
from collections import OrderedDict
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    #setup font
    plt.rcParams['font.family'] = 'Arial'

    # Set global default text sizes
    plt.rcParams['font.size'] = 16  # Default text size
    plt.rcParams['axes.labelsize'] = 20  # Axes label size
    plt.rcParams['xtick.labelsize'] = 16  # X-tick label size
    plt.rcParams['ytick.labelsize'] = 16  # Y-tick label size
    plt.rcParams['legend.fontsize'] = 16  # Legend font size
    plt.rcParams['figure.titlesize'] = 20  # Title size for the whole figure
    plt.rcParams['axes.titlesize'] = 20 # Title size for individual axes

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def classification(index):
    phase = ['SG','DG','SD','DD','SP','DP','HEX','DIS']
    if index>len(phase)-1:
        return None
    else:
        return phase[index]

def list_folders(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def custom_sort_key(filename):
    # Use regex to extract numbers)
    return tuple(map(int, re.findall(r"(\d+)", filename)))

model = CNNClassifier(8)
# model = ModifiedCNNClassifier(8)

# Load the model weights from the .pt file
model_path = "../../Documents/CNN-for-MD/trained_models/model_soft_0828/fold_9_epoch_140.pt"

state_dict=torch.load(model_path,map_location=torch.device('cpu'))

#remove the module. prefix when runing on multiple gpus
new_state_dict = OrderedDict()
for k,v in state_dict.items():
    name=k[7:]
    new_state_dict[name]=v

#load the weight into the model
model.load_state_dict(new_state_dict)


# Set the model in evaluation mode
model.eval()

#set up the y_true and y_pred
y_true=[]
y_pred=[]


criterion = torch.nn.CrossEntropyLoss()

#INPUT
data_path="../../Documents/CNN-for-MD/test_data/test_100_scft/"
folders=list_folders(data_path)


labels=['SG','DG','SD','DD','SP','DP','HEX','DIS']
total_loss = 0.0
count = 0


with torch.no_grad():
    for folder in folders:
        dir=os.path.join(data_path,folder)
        all_file=os.listdir(dir)
        all_file.sort(key=custom_sort_key)
        phase_true=str(folder)
        for file in all_file:
            if 'npy' in file:
                print(file)
                y_true.append(phase_true)
                temp=torch.from_numpy(np.load(dir+'/'+file))
                temp = temp.unsqueeze(0).unsqueeze(0).to(torch.float32)
                t=model.forward(temp)
                print(t.shape)

                # Compute loss
                phase_true_index = labels.index(phase_true)  # Convert string label to index
                loss = criterion(t, torch.tensor([phase_true_index]))
                print("Loss:", loss.item())
                # Update total_loss and count
                total_loss += loss.item()
                count += 1

                probability = F.softmax(t,dim=1)
                round_probability = torch.round(probability*100)/100
                print(round_probability)
                phase_pred=classification(int(torch.argmax(probability)))
                y_pred.append(phase_pred)

# Compute and print the average loss
average_loss = total_loss / count if count > 0 else 0
print("Average Loss:", average_loss)

plot_confusion_matrix(y_true, y_pred, labels)





