import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_IMAGE = 5
input_size = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess_image = A.Compose(
    [
        A.SmallestMaxSize(max_size=input_size + 48),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=input_size, width=input_size),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ]
)

transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
# Preprocess the images and extract image vectors
def val_transform(image_path):
    image = Image.open(image_path).convert("RGB")
    # Preprocessing transforms
    image = transform(image)

    return image.cpu().numpy()


def train_trainsform(image_path):
    # use cv2 for albumentation
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if preprocess_image:
        # img = self.transform(img)
        image = preprocess_image(image=image)["image"]
    return image.cpu().numpy()

def _preprocess_train(folder, num_images, phase_number, saved_dir):
    # Load a pretrained ResNet-50 model
    print(f"Start to preprocess train phase {phase_number}")

    # # Initialize an empty NumPy array to store the data
    data = []
    labels = []
    for cl in range(0 + (phase_number - 1) * 10, (phase_number - 1) * 10 + 10):
        data_dir = folder + f"{cl:03d}/"
        for k in range(0, num_images):
            image_filename = f"{k:03d}.jpg"
            image_path = os.path.join(data_dir, image_filename)

            # Check if the image file exists
            if os.path.exists(image_path):
                # Preprocess the image and extract image vector
                image_vector = train_trainsform(image_path)
                # Store the image vector in the data array
                data.append(image_vector)
                labels.append(cl)
    np.save(f"{saved_dir}/phase_{phase_number}.npy", data)
    np.save(f"{saved_dir}/label_phase_{phase_number}.npy", labels)


def preprocess_train():
    for phase_num in range(1, 11):
        folder = f"data/Train/phase_{phase_num}/"
        saved_dir = "data/preprocessed_train/"

        _preprocess_train(
            folder=folder, num_images=30, phase_number=phase_num, saved_dir=saved_dir
        )
        print(f"Saved preprocessed train phase {phase_num}.")


def preprocess_val():
    print("Start to preprocess validation ...")

    data_dir = "data/Val/"
    saved_dir = "data/preprocessed_val"

    num_images = 887
    # # Initialize an empty NumPy array to store the data
    data = []
    for k in range(0, num_images):
        image_filename = f"{k:03d}.jpg"
        image_path = os.path.join(data_dir, image_filename)

        # Check if the image file exists
        if os.path.exists(image_path):
            # Preprocess the image and extract image vector
            image_vector = val_transform(image_path)
            # Store the image vector in the data array
            data.append(image_vector)
    np.save(f"{saved_dir}/validation.npy", data)
    print("Saved validation.")


preprocess_train()
preprocess_val()
# preprocessed_data = np.load(f'data/preprocessed_val/validation.npy')
# print(len(preprocessed_data))
