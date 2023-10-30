import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from augmentation.data_augmentation import field_augmentation
from torch.optim.lr_scheduler import StepLR
from concurrent.futures import ThreadPoolExecutor
from memory_profiler import profile

class AugmentedDataset(Dataset):
    def __init__(self, data, labels, pre_augmented=False):
        self.data = data
        self.labels = labels
        self.pre_augmented = pre_augmented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        label = self.labels[idx]

        return data_item, label
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=64):
        super(CNNClassifier, self).__init__()
        # Define convolutional layers
        # 1 input channel, 32 output channels
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3))
        self.bn1 = nn.BatchNorm3d(16)

        # 32 input channel, 64 output channels
        self.conv2 = nn.Conv3d(16, 64, kernel_size=(3, 3, 3))
        self.bn2 = nn.BatchNorm3d(64)

        # Adding one more convolutional layer - 64 input channels, 128 output channels
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3))
        self.bn3 = nn.BatchNorm3d(128)

        dim1, dim2, dim3 = self._get_conv_dim_after_3conv(input_dim)
        self.conv_out_size = dim1 * dim2 * dim3 * 128
        # Max pooling layer
        self.pool = nn.MaxPool3d((2, 2, 2))


        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_out_size, 128)
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_dim_after_3conv(self, input_dim, kernel_size=3, pool_stride=2):
        # This function calculates the size after three conv and pool operations
        dim1 = (input_dim - kernel_size + 1) // pool_stride
        dim2 = (32 - kernel_size + 1) // pool_stride  # Set this to your actual input sizes for other dimensions
        dim3 = (32 - kernel_size + 1) // pool_stride  # Set this to your actual input sizes for other dimensions
        for _ in range(2):  # Repeat the operations for conv2 and conv3
            dim1 = (dim1 - kernel_size + 1) // pool_stride
            dim2 = (dim2 - kernel_size + 1) // pool_stride
            dim3 = (dim3 - kernel_size + 1) // pool_stride
        return dim1, dim2, dim3

    def forward(self, x):
        # print("Size input: ", x.size())
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        # print("Size after conv1 and pool: ", x.size())
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        # print("Size after conv2 and pool: ", x.size())
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        # print("Size after conv3 and pool: ", x.size())
        # print("Expected conv_out_size: ", self.conv_out_size)
        x = x.view(-1, self.conv_out_size)
        x = F.leaky_relu(self.fc_bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.fc2(x)

        return x


def train_model(train_loader, model, criterion, optimizer, device):
    # Training loop
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def validate_model(val_loader, model, criterion, device):
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)


            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    return val_loss / len(val_loader), accuracy, cm

def plot_and_save_confusion_matrix(cm, labels_dict, save_path):
    plt.figure(figsize=(10, 7))
    label_names = sorted(labels_dict, key=labels_dict.get)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Saves the figure to the specified path
    plt.close()

def generate_disordered_state(shape=(32, 32, 32), density_range=(0.2, 0.45), noise_level=0.05):
    # Initialize 3D array with average density
    avg_density = np.random.uniform(density_range[0], density_range[1])
    # avg_density = 0.25
    arr = np.full(shape, avg_density)

    # Add localized segregations
    num_regions = np.random.randint(10, 50)  # Random number of regions
    for _ in range(num_regions):
        x, y, z = np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[
            2])
        # dx, dy, dz = np.random.randint(2, 7), np.random.randint(2, 7), np.random.randint(2,7)
        dx, dy, dz = np.random.randint(5, 15), np.random.randint(5, 15), np.random.randint(5, 15)
        segregation_density = np.random.uniform(avg_density, 1.)  # Random density for this region
        x_range = np.arange(x, x + dx) % shape[0]
        y_range = np.arange(y, y + dy) % shape[1]
        z_range = np.arange(z, z + dz) % shape[2]

        arr[np.ix_(x_range, y_range, z_range)] = segregation_density
        # print(dx, dy, dz, segregation_density)
    # Add random noise
    noise = np.random.uniform(-noise_level, noise_level, size=shape)
    arr += noise
    arr = arr.reshape(32, 32, 32)
    return arr

def normalize_numpy_array(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

@profile
def process_sample(i, data, labels):
    current_label = labels[i]
    if current_label == 7:
        generated_disorder = generate_disordered_state()
        weights_gen_disorder = np.random.uniform(0.5, 0.7)
        num_to_mix = np.random.choice([2, 3])
        selected_classes = np.random.choice(np.unique(labels[labels != 7]), num_to_mix, replace=False)
        selected_samples = [data[np.random.choice(np.where(labels == c)[0])].cpu().numpy().squeeze(0) for c in
                            selected_classes]
        augmented_samples = []
        for sample in selected_samples:
            augmenter = field_augmentation(duplicate=True, crop=False, rotation=True,
                                           translation=True, rand_noise=True, inver_intensity=False,
                                           rand_flip=True, contrast=False)
            augmented_data = augmenter.self_augmentation(sample)
            augmented_data = augmenter.vol_augmentation()(image=augmented_data)["image"]
            augmented_samples.append(augmented_data)
        weights_others = np.random.uniform(0.2, 0.3, num_to_mix)
        weights_others /= np.sum(weights_others)
        weights_others *= (1 - weights_gen_disorder)
        # Compute weighted sum to get a disordered state
        disordered_state = weights_gen_disorder * generated_disorder + sum(
            w * s for w, s in zip(weights_others, augmented_samples))
        disordered_state = normalize_numpy_array(disordered_state)
        # disordered_state = disordered_state.reshape(32,32,32)
        fourier_data = np.fft.fftn(disordered_state)
        fourier_data = np.real(fourier_data)
        fourier_data = normalize_numpy_array(fourier_data)
        combined_data = np.concatenate((disordered_state, fourier_data), axis=0)
        return torch.from_numpy(combined_data).unsqueeze(0)
    else:
        numpy_data = data[i].cpu().numpy().squeeze(0)
        augmenter = field_augmentation(duplicate=True, crop=False, rotation=False,
                                       translation=True, rand_noise=True, inver_intensity=False,
                                       rand_flip=True, contrast=False)
        augmented_data = augmenter.self_augmentation(numpy_data)
        augmented_data = augmenter.vol_augmentation()(image=augmented_data)["image"]
        fourier_data = np.fft.fftn(augmented_data)
        fourier_data = np.real(fourier_data)
        fourier_data = normalize_numpy_array(fourier_data)
        augmenter = field_augmentation(duplicate=False, crop=False, rotation=True,
                                       translation=False, rand_noise=False, inver_intensity=False,
                                       rand_flip=False, contrast=False)
        augmented_data = augmenter.vol_augmentation()(image=augmented_data)["image"]
        augmented_data = normalize_numpy_array(augmented_data)
        combined_data = np.concatenate((augmented_data, fourier_data), axis=0)

        return torch.from_numpy(combined_data).unsqueeze(0)
@profile
def pre_augment_data(data, labels):
    with ThreadPoolExecutor(max_workers=4) as executor:
        augmented_data_list = list(executor.map(process_sample, range(len(data)), [data]*len(data), [labels]*len(data)))

    augmented_data_tensor = torch.stack(augmented_data_list, dim=0).to(torch.float32)
    return augmented_data_tensor, labels

def train_cross_validate_and_save(data, labels, label_dict, k, epochs, parent_folder, batch_size, save_freq, device):

    # Create separate folders for losses, confusion matrix images, and model weights.
    loss_folder = os.path.join(parent_folder, 'losses')
    cm_folder = os.path.join(parent_folder, 'confusion_matrices')
    weights_folder = os.path.join(parent_folder, 'weights')

    os.makedirs(loss_folder, exist_ok=True)
    os.makedirs(cm_folder, exist_ok=True)
    os.makedirs(weights_folder, exist_ok=True)

    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=99)
    validation_scores = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data, labels)):
        print(f"Training Fold {fold + 1}/{k}")
        #
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
        model = CNNClassifier(num_classes=args.num_classes).to(device)  # Assuming 5 classes
        if (device.type == 'cuda') and (args.ngpu > 1):
            model = nn.DataParallel(model, list(range(args.ngpu)))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

        best_val_acc = 0.0
        train_losses = []
        val_losses = []

        augmented_data_v, augmented_labels_v = pre_augment_data(data[val_ids], labels[val_ids])
        augmented_val_dataset = AugmentedDataset(augmented_data_v, augmented_labels_v)
        val_loader = DataLoader(augmented_val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            augmented_data, augmented_labels = pre_augment_data(data[train_ids], labels[train_ids])
            augmented_train_dataset = AugmentedDataset(augmented_data, augmented_labels)
            train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
            train_loss = train_model(train_loader, model, criterion, optimizer, device)
            scheduler.step()

            val_loss, val_acc, cm = validate_model(val_loader, model, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save confusion matrix image
            plot_and_save_confusion_matrix(cm, label_dict,
                                           os.path.join(cm_folder, f"fold_{fold + 1}_epoch_{epoch + 1}.png"))

            print(f"Epoch {epoch + 1}/{epochs} - Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            # Save model weights
            if (epoch + 1) % save_freq == 0:
                torch.save(model.state_dict(), os.path.join(weights_folder, f"fold_{fold + 1}_epoch_{epoch + 1}.pt"))

            # Optional: Save if there's an improvement in validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(weights_folder, f"fold_{fold + 1}_best_model.pt"))

        # Save the training and validation losses for this fold to a txt file
        with open(os.path.join(loss_folder, f"fold_{fold + 1}_train_losses.txt"), 'w') as f:
            for loss in train_losses:
                f.write(f"{loss}\n")

        with open(os.path.join(loss_folder, f"fold_{fold + 1}_val_losses.txt"), 'w') as f:
            for loss in val_losses:
                f.write(f"{loss}\n")

        validation_scores.append(best_val_acc)
        print(f"Best validation accuracy for fold {fold + 1}: {best_val_acc:.2f}%")

    print(f"Average Validation Accuracy: {np.mean(validation_scores):.2f}% ± {np.std(validation_scores):.2f}%")

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Train a 3D CNN on given dataset with cross-validation")

    # Adding the arguments
    parser.add_argument('--dataroot', type=str, default="./data/",
                        help="Path to the dataset")

    parser.add_argument('--num_classes', type=int, default=8,
                        help="Number of classes in the dataset")

    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")

    parser.add_argument('--batch_size', type=int, default=64,
                        help="Size of mini-batch for training")

    parser.add_argument('--k_folds', type=int, default=10,
                        help="Number of folds for cross-validation")

    parser.add_argument('--output_dir', type=str, default='./model_checkpoints_test',
                        help="Directory to save the model weights, loss values and confusion matrices")

    parser.add_argument('--save_freq', type=int, default=1,
                        help="Number of epochs after which model weights should be saved")

    parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')


    # Parse the arguments
    args = parser.parse_args()

    # Now we use the passed arguments
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    data = torch.load(os.path.join(args.dataroot, "new_extended_data_2.pt"))
    labels = torch.load(os.path.join(args.dataroot, "new_extended_labels_2.pt"))
    label_dict = json.load(open(os.path.join(args.dataroot, "label_dict.json")))

    # Adjust the train_cross_validate_and_save function to use arguments
    train_cross_validate_and_save(data, labels, label_dict,
                                    k=args.k_folds,
                                    epochs=args.epochs,
                                    parent_folder=args.output_dir,
                                    batch_size=args.batch_size,
                                    save_freq=args.save_freq,
                                    device=device)


#
# def pre_augment_data(data, labels):
#     augmented_data_tensor = torch.empty((data.shape[0], 1, 64, 32, 32))
#     for i in range(len(data)):
#         current_label = labels[i]
#         if current_label == 7:
#             generated_disorder = generate_disordered_state()
#             weights_gen_disorder = np.random.uniform(0.5, 0.7)
#             num_to_mix = np.random.choice([2, 3])
#             selected_classes = np.random.choice(np.unique(labels[labels != 7]), num_to_mix, replace=False)
#             selected_samples = [data[np.random.choice(np.where(labels == c)[0])].cpu().numpy().squeeze(0) for c in
#                                 selected_classes]
#             augmented_samples = []
#             for sample in selected_samples:
#                 augmenter = field_augmentation(duplicate=True, crop=False, rotation=True,
#                                                translation=True, rand_noise=True, inver_intensity=False,
#                                                rand_flip=True, contrast=False)
#                 augmented_data = augmenter.self_augmentation(sample)
#                 augmented_data = augmenter.vol_augmentation()(image=augmented_data)["image"]
#                 augmented_samples.append(augmented_data)
#             weights_others = np.random.uniform(0.2, 0.3, num_to_mix)
#             weights_others /= np.sum(weights_others)
#             weights_others *= (1 - weights_gen_disorder)
#             # Compute weighted sum to get a disordered state
#             disordered_state = weights_gen_disorder * generated_disorder + sum(
#                 w * s for w, s in zip(weights_others, augmented_samples))
#             disordered_state = normalize_numpy_array(disordered_state)
#             # disordered_state = disordered_state.reshape(32,32,32)
#             fourier_data = np.fft.fftn(disordered_state)
#             fourier_data_real = np.real(fourier_data)
#             fourier_data_real = normalize_numpy_array(fourier_data_real)
#             combined_data = np.concatenate((disordered_state, fourier_data_real), axis=0)
#             augmented_data_tensor[i][0] = torch.from_numpy(combined_data)
#         else:
#             numpy_data = data[i].cpu().numpy().squeeze(0)
#             augmenter = field_augmentation(duplicate=True, crop=False, rotation=False,
#                                            translation=True, rand_noise=True, inver_intensity=False,
#                                            rand_flip=True, contrast=False)
#             augmented_data = augmenter.self_augmentation(numpy_data)
#             augmented_data = augmenter.vol_augmentation()(image=augmented_data)["image"]
#             fourier_data = np.fft.fftn(augmented_data)
#             fourier_data_real = np.real(fourier_data)
#             fourier_data_real = normalize_numpy_array(fourier_data_real)
#             augmenter = field_augmentation(duplicate=False, crop=False, rotation=True,
#                                            translation=False, rand_noise=False, inver_intensity=False,
#                                            rand_flip=False, contrast=False)
#             augmented_data = augmenter.vol_augmentation()(image=augmented_data)["image"]
#             augmented_data = normalize_numpy_array(augmented_data)
#             combined_data = np.concatenate((augmented_data, fourier_data_real), axis=0)
#             augmented_data_tensor[i][0] = torch.from_numpy(combined_data)
#
#     return augmented_data_tensor, labels