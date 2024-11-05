import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

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
        img_filename = self.dataframe.iloc[idx]['image_name']
        mask_filename = self.dataframe.iloc[idx]['mask_name']

        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return image, mask, img_filename, mask_filename

# Example usage
# Initialize the dataset
image_dir = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'
mask_dir = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz-msk/'

# Instantiate the dataset
dataset = SegmentationDataset(image_dir, mask_dir)

# Extract the DataFrame with filenames
filenames_df = dataset.dataframe

# Save the DataFrame to a CSV file
csv_path = 'filenames.csv'
filenames_df.to_csv(csv_path, index=False)
print(f"Filenames saved to {csv_path}")

