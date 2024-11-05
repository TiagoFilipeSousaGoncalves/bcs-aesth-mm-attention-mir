import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import re
import numpy as np
from torchvision.utils import save_image
import cv2



def get_filenames(image_dir, mask_dir, valid_filenames_set):
    # Get all the filenames in the image directory
    image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    # Generate mask filenames based on the image filenames and check if the mask file exists
    valid_pairs = []
    for image_file in image_filenames:
        if image_file in valid_filenames_set:  # Check if the image is in the valid filenames set
            mask_file = image_file.replace('_resized', '_mask_resized')  # Adjust this based on your naming convention
            if os.path.isfile(os.path.join(mask_dir, mask_file)):
                valid_pairs.append({'image_name': image_file, 'mask_name': mask_file})
    
    return pd.DataFrame(valid_pairs)

#unnormalized a tensor image
def unnormalize_tensor(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Define the function to save the numpy array as an image
def save_image_from_array(array, file_name):
    array = array.astype(np.uint8)  # Ensure array is of type uint8
    if array.ndim == 2:
        array = np.stack((array,) * 3, axis=-1)  # Make grayscale to RGB
    Image.fromarray(array).save(file_name)

#test for segmentation - mask generation
def test_model(model, dataloader, mse_loss, device, save_dir='/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/', cropped_save_dir='/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/Cropped_images/', save_every_n_batches=1):
        model.eval() 
        test_dice_scores = []
        test_mse_scores = [] 

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
                        mask_save_path = os.path.join(save_dir, f'predicted_{mask_filenames[j]}')
            
                        # Save the original image
                        save_image(data[j].cpu(), img_save_path)

                        # Convert predicted mask to binary mask and save
                        binary_mask = (predictions[j].cpu().numpy().squeeze() * 255).astype(np.uint8)
                        binary_mask_img = Image.fromarray(binary_mask)
                        binary_mask_img.save(mask_save_path)

                        # Bounding boxes and extract features using ResNet
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        unnorm_image = unnormalize_tensor(data[j].cpu().clone(), mean=[0.4506, 0.4514, 0.4939], std=[0.2421, 0.1717, 0.1641])
                        original_image = unnorm_image.numpy().transpose(1, 2, 0)
                        original_image = (original_image * 255).astype(np.uint8) 
                        image_with_boxes = original_image.copy()

                        if contours:
                            # Find the minimum and maximum coordinates for all contours
                            min_x = min_y = float('inf')
                            max_x = max_y = float('-inf')

                            for contour in contours:
                                if cv2.contourArea(contour) > 200:
                                    x, y, w, h = cv2.boundingRect(contour)
                                    min_x = min(min_x, x)
                                    min_y = min(min_y, y)
                                    max_x = max(max_x, x + w)
                                    max_y = max(max_y, y + h)

                            # Draw a single bounding box that encompasses all contours
                            cv2.rectangle(image_with_boxes, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                            # Crop the image using the bounding box coordinates
                            cropped_image = original_image[min_y:max_y, min_x:max_x]

                            # Save the cropped image - use it to save generated images
                            #cropped_img_save_path = os.path.join(cropped_save_dir, f'{img_filenames[j]}')
                            #cv2.imwrite(cropped_img_save_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                        # Save the image with bounding boxes
                       # image_with_boxes_save_path = os.path.join(save_dir, f'boxed_{img_filenames[j]}')
                       # cv2.imwrite(image_with_boxes_save_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

        # Calculate and print the average test MSE and Dice scores
        avg_test_dice = np.mean(test_dice_scores)
        avg_test_mse = np.mean(test_mse_scores)
        print(f"Average Test Dice Score: {avg_test_dice:.4f}")
        print(f"Average Test MSE: {avg_test_mse:.4f}")

        return avg_test_dice, avg_test_mse

class SegmentationDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, image_transforms=None, mask_transforms=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_filename = self.dataframe.iloc[idx]['image_name']
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
            torch.manual_seed(seed)
            mask = self.mask_transforms(mask)

        return image, mask, img_filename, mask_filename

    def set_transforms(self, image_transforms, mask_transforms):
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms


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

# Define the function to extract numbers based on the conditions
def extract_numbers(image_name):    
    # Check if the image_name starts with a number
    match = re.match(r'^(\d+)_', image_name)
    if match:
        return match.group(1)
    else:
        # Extract numbers after the first "_" until the next "_"
        match = re.search(r'_(\d+)_', image_name)
        if match:
            return match.group(1)
    return None

def extract_name(image_name, directory_path):
    # Remove the directory path from the image_name
    if image_name.startswith(directory_path):
        return image_name[len(directory_path):]
    return image_name

# Function to extract features
def extract_features(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

#retrival evaluation metrics
def precision_at_k(y_true, y_pred, k):
    y_true_set = set(y_true)
    y_pred_set = set(y_pred[:k])
    return len(y_true_set & y_pred_set) / k

def recall_at_k(y_true, y_pred, k):
    y_true_set = set(y_true)
    y_pred_set = set(y_pred[:k])
    return len(y_true_set & y_pred_set) / len(y_true_set)

def dcg_at_k(y_true, y_pred, k):
    y_true_set = set(y_true)
    dcg = 0.0
    for i, p in enumerate(y_pred[:k]):
        if p in y_true_set:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def ndcg_at_k(y_true, y_pred, k):
    y_true_set = set(y_true)
    dcg = sum([1/np.log2(i+2) if y_pred[i] in y_true_set else 0 for i in range(k)])
    idcg = sum([1/np.log2(i+2) for i in range(min(len(y_true_set), k))])
    return dcg/idcg

def ensure_list_format(value):
    if isinstance(value, str):
        return list(map(int, value.strip('[]').split(',')))
    elif isinstance(value, list):
        return value
    else:
        return []

def images_path(images):
    if images == 'original_images':
        path = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'
    elif  images == 'cropped_save_dir':
        path = '/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/Cropped_images/'
    else :
        raise ValueError("Invalid images type name. Choose either 'original_images'or 'cropped_save_dir'")
    return path


