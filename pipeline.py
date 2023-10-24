import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from model import (
    BirdClassificationModel,
    BirdDataset,
    BirdValidationDataset,
    DistillationLoss,
)
from torchvision import transforms, models
import warnings
warnings.filterwarnings("ignore")

import yaml

# Load the YAML file
with open('configs.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")  # Use CPU

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=10)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)
parser.add_argument("-l", "--lr", type=int, help="Learning rate", default=0.01)

# Parse the arguments
args = parser.parse_args()
print(device)


def train_model(
    model,
    train_loader,
    validation_data,
    validation_labels,
    teacher_model, 
    num_epochs,
    batch_size,
    lr
):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = DistillationLoss() if teacher_model else nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss = criterion(outputs, labels, teacher_outputs)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        validation_dataset = BirdValidationDataset(validation_data)
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )

        predictions = evaluate_model(model, validation_loader)
        val_acc = accuracy_score(validation_labels, predictions)
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, train_Accuracy: {accuracy * 100:.2f}%, val_Accuracy: {val_acc * 100:.2f}%"
        )
    return model


def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []  # To store translated predictions

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            _, predicted = torch.max(
                output, 1
            )  # Get the index of the max log-probability
            all_predictions.extend(predicted.cpu().detach().numpy())

    all_predictions = np.array(all_predictions)
    all_predictions = all_predictions.astype(int)
    return all_predictions


def train_phase():
    data_dir = "data/train_val_phase/"
    teacher_model = None

    # Define the optimizer and loss function

    for phase_num in range(1, 11):
        num_class = phase_num * 10
        num_epochs = config["num_epoch"][f"phase_{phase_num}"]
        batch_size = config["batch_size"][f"phase_{phase_num}"]
        lr = config["lr"][f"phase_{phase_num}"]

        print("phase: ", phase_num, "classes: ", num_class, "num_epochs: ", num_epochs, "batch: ", batch_size, "lr: ", lr)

        # Load the train
        preprocessed_data = np.load(data_dir + f"train_phase_{phase_num}.npy")
        # Load the labels
        labels = np.load(data_dir + f"train_label_phase_{phase_num}.npy")

        # Convert the NumPy array to PyTorch tensors
        data = torch.from_numpy(preprocessed_data).float()
        labels = torch.from_numpy(labels).long()

        # Define the dataset and dataloader
        dataset = BirdDataset(data, labels)

        validation_data = np.load(data_dir + f"val_phase_{phase_num}.npy")
        validation_data = torch.from_numpy(validation_data).float()
        val_labels = np.load(data_dir + f"val_label_phase_{phase_num}.npy")
        # Ensure shuffle = False when evaluating on validation and test
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = models.resnet18(pretrained=True)
        # model.dropout = nn.Dropout(0)
        model.fc = nn.Linear(model.fc.in_features, num_class)
        teacher_model.fc = nn.Linear(model.fc.in_features, num_class)
        model.to(device)

        if teacher_model:
            new_state_dict = model.state_dict()
            old_state_dict = {
                name: param
                for name, param in model.state_dict().items()
                if name in new_state_dict and new_state_dict[name].shape == param.shape
            }
            new_state_dict.update(old_state_dict)
            model.load_state_dict(new_state_dict, strict=False)
        model = train_model(
            model,
            train_loader,
            validation_data,
            val_labels,
            teacher_model,
            num_epochs,
            batch_size,
            lr
        )

        teacher_model = model


train_phase()
