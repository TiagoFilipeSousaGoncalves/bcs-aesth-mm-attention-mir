import os
import pandas as pd
from PIL import Image
import numpy as np
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset
import torchvision.transforms
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from collections import deque
import cv2


def get_filenames(image_dir, mask_dir):
    # Get all the filenames in the image directory
    image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    # Generate mask filenames based on the image filenames and check if the mask file exists
    valid_pairs = []
    for image_file in image_filenames:
        mask_file = image_file  # Adjust this based on your naming convention
        if os.path.isfile(os.path.join(mask_dir, mask_file)):
            valid_pairs.append({'image_name': image_file, 'mask_name': mask_file})
    
    return pd.DataFrame(valid_pairs)


# Define your transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4506, 0.4514, 0.4939], std=[0.2421, 0.1717, 0.1641]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
])

# Define your transformations for images and masks
val_image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4506, 0.4514, 0.4939], std=[0.2421, 0.1717, 0.1641]),
])

val_mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


device = torch.device("cuda:0")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transforms=None, mask_transforms=None):
        self.dataframe = get_filenames(image_dir, mask_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_filename = self.dataframe.iloc[idx]['image_name'] #added to get the file names/id
        mask_filename = self.dataframe.iloc[idx]['mask_name']

        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_transforms is not None:
            seed = torch.random.initial_seed()
            torch.manual_seed(seed)
            image = self.image_transforms(image)

        if self.mask_transforms is not None:
            torch.manual_seed(seed)  # Use the same seed to ensure identical transformations
            mask = self.mask_transforms(mask)

        return image, mask, img_filename, mask_filename


    def set_transforms(self, image_transforms, mask_transforms):
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

# Initialize the dataset
image_dir = '/nas-ctm01/datasets/private/CINDERELLA/american_dataset/images/0413/'
mask_dir = '/nas-ctm01/datasets/private/CINDERELLA/american_dataset/segmentations/0413/'

dataset = SegmentationDataset(
    image_dir=image_dir, 
    mask_dir=mask_dir, 
    image_transforms=image_transform,
    mask_transforms=mask_transform     
)



# Initialize the datasets
train_dataset = SegmentationDataset(
    image_dir=image_dir, 
    mask_dir=mask_dir, 
    image_transforms=image_transform,  # Training transformations
    mask_transforms=mask_transform
)

validation_dataset = SegmentationDataset(
    image_dir=image_dir, 
    mask_dir=mask_dir, 
    image_transforms=val_image_transform,  # Validation transformations
    mask_transforms=val_mask_transform
)

test_dataset = SegmentationDataset(
    image_dir=image_dir, 
    mask_dir=mask_dir, 
    image_transforms=val_image_transform,  # Usually, the same as validation
    mask_transforms=val_mask_transform
)


torch.manual_seed(14)

# Assuming 'dataset' is your complete dataset
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)  # 70% for training
validation_size = int(0.15 * dataset_size)  # 15% for validation
test_size = dataset_size - train_size - validation_size  # Remaining 15% for testing


# Split the dataset based on indices instead of reinitializing
indices = torch.randperm(len(train_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + validation_size]
test_indices = indices[train_size + validation_size:]

train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
validation_dataset = torch.utils.data.Subset(validation_dataset, val_indices)
test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

""" To check the dataset size"""
"""
print(f"Total dataset size: {len(dataset)}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(validation_dataset)}")
print(f"Test set size: {len(test_dataset)}")
"""

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


def double_conv(in_channels, out_channels, dropout=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),  # Add Batch Normalization here
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),  # And here
        nn.ReLU(inplace=True)
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(UNet, self).__init__()
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512, dropout=dropout_rate)

        self.bottleneck = double_conv(512, 1024, dropout=dropout_rate)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = double_conv(1024 + 256, 256, dropout=dropout_rate)
        self.conv_up2 = double_conv(256 + 128, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.conv_down4(x)

        bottleneck = self.bottleneck(conv4)
        x = self.upsample(bottleneck)

        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)
        out = self.last_conv(x)
        out = torch.sigmoid(out)
        return out
    

def dice_coef_loss(inputs, target):
    smooth = 1.0
    inputs_flat = inputs.view(inputs.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
    inputs = inputs.float()
    target = target.float()
    
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore

def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Increases the learning rate from a small value to
    the originally set value, over a few warm-up iterations.

    :param optimizer: Optimizer whose learning rate needs to be scheduled.
    :param warmup_iters: Number of iterations over which the learning rate will be linearly increased.
    :param warmup_factor: Starting factor of the learning rate (as a fraction of the original lr).
    """

    def lr_lambda(iter):
        # Calculate the learning rate factor
        if iter < warmup_iters:
            alpha = float(iter) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        return 1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Define the function to save the numpy array as an image
def save_image_from_array(array, file_name):
    array = array.astype(np.uint8)  # Ensure array is of type uint8
    if array.ndim == 2:
        array = np.stack((array,) * 3, axis=-1)  # Make grayscale to RGB
    Image.fromarray(array).save(file_name)

    
def log_output_stats(outputs):
    print("Output Stats:")
    print(f"Max value: {outputs.max().item()}")
    print(f"Min value: {outputs.min().item()}")
    print(f"Mean value: {outputs.mean().item()}")
    print(f"Standard Deviation: {outputs.std().item()}")

def threshold_check(outputs, threshold=0.6):
    thresholded_outputs = (outputs > threshold).float()
    active_pixels = thresholded_outputs.sum().item()
    total_pixels = np.prod(thresholded_outputs.shape)
    print(f"Active pixels: {active_pixels}, Total pixels: {total_pixels}, Percentage: {(active_pixels / total_pixels) * 100:.2f}%")

def validate_model(model, dataloader, mse_loss):
    model.eval()  # Set model to evaluation mode
    val_dice_scores = []
    val_mse_scores = []  # Track MSE scores during validation
    with torch.no_grad():
        for data, target, img_filenames, mask_filenames in dataloader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            
            # Log output statistics and check thresholds
            #log_output_stats(outputs)
            #threshold_check(outputs)  # Check how many outputs are above the threshold

            predictions = (outputs > 0.6).float()  # Threshold predictions
            dice_score = dice_coef_metric(predictions, target)
            mse_score = mse_loss(outputs, target.float())  # Compute MSE

            val_dice_scores.append(dice_score.item())
            val_mse_scores.append(mse_score.item())

    return np.mean(val_dice_scores), np.mean(val_mse_scores)


def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs, patience=5):
    print(model_name)
    loss_history = []
    train_history = []
    val_history = []
    mse_history = []

    # Early stopping initialization
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        losses = []
        train_dice_scores = []
        mse_scores = []

        for i_step, (data, target, img_filenames, mask_filenames) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = train_loss(outputs, target)

            outputs = torch.sigmoid(outputs)  # Assumes the output of your model are logits
            predictions = (outputs > 0.6).float()

            # Calculate Dice score and MSE
            dice_score = dice_coef_metric(predictions, target)
            mse_score = mse_loss(outputs, target.float())

            train_dice_scores.append(dice_score.item())
            mse_scores.append(mse_score.item())

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        # Validation
        val_mean_iou, val_mean_mse = validate_model(model, val_loader, mse_loss)
        print(f"Epoch [{epoch+1}]: Mean Loss: {np.mean(losses):.4f}, Mean Train Dice: {np.mean(train_dice_scores):.4f}, Mean Train MSE: {np.mean(mse_scores):.4f}, Mean Val Dice: {val_mean_iou:.4f}, Mean Val MSE: {val_mean_mse:.4f}")
        
        # Early stopping logic
        current_val_loss = np.mean(losses)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1  # Increment counter if no improvement

        # Check if patience has run out
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch+1} due to no improvement in validation loss.")
            break  # Break out of the loop to stop training
        
        loss_history.append(np.mean(losses))
        train_history.append(np.mean(train_dice_scores))
        val_history.append(val_mean_iou)
        mse_history.append(np.mean(mse_scores))


    # Load the best model state if early stopping was triggered
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return loss_history, train_history, val_history, mse_history



def test_model(model, dataloader, mse_loss, device, save_dir='/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/american_dataset/', save_every_n_batches=5):
    model.eval()  # Set the model to evaluation mode
    test_dice_scores = []
    test_mse_scores = []  # Track MSE scores during testing

    with torch.no_grad():
        for i, (data, target, img_filenames, mask_filenames) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities

            predictions = (outputs > 0.6).float()  # Threshold predictions
            dice_score = dice_coef_metric(predictions, target)
            mse_score = mse_loss(outputs, target.float())

            test_dice_scores.append(dice_score.item())
            test_mse_scores.append(mse_score.item())

            # Save images and masks every n batches
            if (i + 1) % save_every_n_batches == 0:
                for j in range(data.size(0)):
                    # Prepare paths
                    img_save_path = os.path.join(save_dir, f'original_{img_filenames[j]}')
                    bbox_save_path = os.path.join(save_dir, f'predicted_{mask_filenames[j]}')

                    # Save the original image
                    save_image(data[j].cpu(), img_save_path)

                    # Convert predicted mask to binary mask and save
                    binary_mask = (predictions[j].cpu().numpy().squeeze() * 255).astype(np.uint8)
                    binary_mask_img = Image.fromarray(binary_mask)
                    binary_mask_img.save(bbox_save_path)

                    # Generate and save bounding boxes
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    image_with_boxes = cv2.cvtColor(np.array(binary_mask_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(bbox_save_path, image_with_boxes)

                    print(f"Saved {img_save_path} and {bbox_save_path}")

    # Calculate and print the average test MSE and Dice scores
    avg_test_dice = np.mean(test_dice_scores)
    avg_test_mse = np.mean(test_mse_scores)
    print(f"Average Test Dice Score: {avg_test_dice:.4f}")
    print(f"Average Test MSE: {avg_test_mse:.4f}")

    return avg_test_dice, avg_test_mse





# Model initialization
model = UNet().to(device)

#MSE Loss
mse_loss = MSELoss()

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-5)

# Optional: Learning Rate Scheduler
# Assuming you've defined warmup_lr_scheduler
lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters=100, warmup_factor=1.0 / 1000)


# Loss function
# Assuming you want to use the combined BCE and Dice loss
criterion = bce_dice_loss

# Number of epochs to train
num_epochs = 120

#patience
patience = 5

# Run the training
loss_history, train_history, val_history, mse_history  = train_model(
    "U-Net Training",
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    patience
)


# Optional: Save your trained model
torch.save(model.state_dict(), 'unet_model_Aaug_american.pth')

# Load the best saved model before testing
best_model_state = torch.load('unet_model_Aaug_american.pth')
model.load_state_dict(best_model_state)
test_dice, test_loss = test_model(model, test_loader, bce_dice_loss, device)

