import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms, models

class BirdDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label


class BirdValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        return image


# # Define the model architecture
class BirdClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BirdClassificationModel, self).__init__()
        self.input_size = input_size
        self.dropout1 = nn.Dropout(0.42)
        self.dropout2 = nn.Dropout(0.04)
        self.model = models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x))) 
        return x


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.1, temperature=1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, teacher_outputs):
        if outputs.shape[1] != teacher_outputs.shape[1]:
            padding = torch.zeros(
                (teacher_outputs.shape[0], outputs.shape[1] - teacher_outputs.shape[1]),
                device=teacher_outputs.device,
            )
            teacher_outputs = torch.cat((teacher_outputs, padding), dim=1)

        loss = (1 - self.alpha) * self.criterion(outputs, labels)
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(outputs / self.temperature, dim=1),
            nn.functional.softmax(teacher_outputs / self.temperature, dim=1),
        )
        loss += self.alpha * distillation_loss
        return loss
