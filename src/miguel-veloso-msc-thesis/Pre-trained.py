import os
import pandas as pd
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def get_filenames(image_dir, mask_dir):
    # Get all the filenames in the image directory
    image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    # Generate mask filenames based on the image filenames and check if the mask file exists
    valid_pairs = []
    for image_file in image_filenames:
        mask_file = image_file.replace('_resized', '_mask_resized')  # Adjust this based on your naming convention
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




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        img_path = os.path.join(self.image_dir, self.dataframe.iloc[idx]['image_name'])
        mask_path = os.path.join(self.mask_dir, self.dataframe.iloc[idx]['mask_name'])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_transforms is not None:
            seed = torch.random.initial_seed()
            torch.manual_seed(seed)
            image = self.image_transforms(image)

        if self.mask_transforms is not None:
            torch.manual_seed(seed)  # Use the same seed to ensure identical transformations
            mask = self.mask_transforms(mask)

        return image, mask


# Initialize the dataset
image_dir = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'
mask_dir = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz-msk/'

dataset = SegmentationDataset(
    image_dir=image_dir, 
    mask_dir=mask_dir, 
    image_transforms=image_transform,
    mask_transforms=mask_transform     
)

torch.manual_seed(14)

# Assuming 'dataset' is your complete dataset
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)  # 70% for training
validation_size = int(0.15 * dataset_size)  # 15% for validation
test_size = dataset_size - train_size - validation_size  # Remaining 15% for testing

# Split the dataset
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

model.to(device)

def dice_coef_loss(inputs, target):
    smooth = 1.0
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


def validate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    val_dice_scores = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            predictions = (outputs > 0.5).float()  # Threshold predictions

            dice_score = dice_coef_metric(predictions, target)
            val_dice_scores.append(dice_score.item())

    return np.mean(val_dice_scores)  # Return the average Dice score


def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs):  
    
    print(model_name)
    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()  # Enter train mode
        
        # We store the training loss and dice scores
        losses = []
        train_iou = []
                
        if lr_scheduler:
            warmup_factor = 1.0 / 100
            warmup_iters = min(100, len(train_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        
        for i_step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
                      
            outputs = model(data)
            
            out_cut = np.copy(outputs.data.cpu().numpy())

            # If the score is less than a threshold (0.5), the prediction is 0, otherwise its 1
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            
            train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            
            loss = train_loss(outputs, target)
            
            losses.append(loss.item())
            train_iou.append(train_dice)

            # Reset the gradients
            optimizer.zero_grad()
            # Perform backpropagation to compute gradients
            loss.backward()
            # Update the parameters with the computed gradients
            optimizer.step()
    
            if lr_scheduler:
                lr_scheduler.step()
        
        val_mean_iou = validate_model(model, val_loader)
        
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)
        
        print("Epoch [%d]" % (epoch))
        print("Mean loss on train:", np.array(losses).mean(), 
              "\nMean DICE on train:", np.array(train_iou).mean(), 
              "\nMean DICE on validation:", val_mean_iou)
        
    return loss_history, train_history, val_history



# Model initialization
#model = UNet().to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Optional: Learning Rate Scheduler
# Assuming you've defined warmup_lr_scheduler
lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters=100, warmup_factor=1.0 / 1000)


# Loss function
# Assuming you want to use the combined BCE and Dice loss
criterion = bce_dice_loss

# Number of epochs to train
num_epochs = 25

# Run the training
loss_history, train_history, val_history = train_model(
    "U-Net Training",
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs
)

# Optional: Save your trained model
torch.save(model.state_dict(), 'unet_model.pth')
