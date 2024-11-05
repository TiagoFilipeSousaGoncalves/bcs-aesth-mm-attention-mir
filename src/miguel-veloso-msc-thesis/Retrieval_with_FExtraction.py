import os
import pandas as pd
from PIL import Image
import numpy as np
import random

import torch

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics.pairwise import euclidean_distances

from Models import UNet, load_model
from Utilities_file import (
    get_filenames, 
    test_model,
    SegmentationDataset, 
    dice_coef_loss, 
    bce_dice_loss, 
    dice_coef_metric, 
    extract_features,
    extract_name, 
    precision_at_k, 
    ndcg_at_k, 
    recall_at_k, 
    dcg_at_k, 
    unnormalize_tensor, 
    save_image_from_array,
    ensure_list_format,
    images_path
)

# Define your transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4506, 0.4514, 0.4939], std=[0.2421, 0.1717, 0.1641]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

device = torch.device("cuda:0")

# Path to the CSV file
base_path = '/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/'
csv_file_1 = base_path + 'df_combined_p.csv'

# Load the DataFrame with image and mask file names
valid_filenames_df = pd.read_csv(csv_file_1)

# Define the directory path to be removed
directory_path = "/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/"

# Apply the extract_name function to the PID column
valid_filenames_df['image_name'] = valid_filenames_df['image PID'].apply(lambda x: extract_name(x, directory_path))

# Save the DataFrame to a CSV file if needed
valid_filenames_df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/check_Validnames.csv', index=False)

valid_filenames_set = set(valid_filenames_df['image_name'])
# Initialize the dataset
image_dir = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'
mask_dir = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz-msk/'

dataframe = get_filenames(image_dir, mask_dir, valid_filenames_set)

dataset = SegmentationDataset(dataframe, image_dir, mask_dir, image_transform, mask_transform)

torch.manual_seed(14)
test_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)    

# Define the unnormalization transform
unnormalize = transforms.Normalize(
    mean=[-0.4506 / 0.2421, -0.4514 / 0.1717, -0.4939 / 0.1641],
    std=[1 / 0.2421, 1 / 0.1717, 1 / 0.1641]
)

# Initialize the model
model = UNet(dropout_rate=0.2).to(device)
# Define a loss function
mse_loss = torch.nn.MSELoss()

# Load the best saved model before testing
model.load_state_dict(torch.load('unet_model.pth'))
test_dice, test_loss = test_model(model, test_loader, bce_dice_loss, device)

############################### Feature Extraction ##################################
####################### Resnet/Densenet Feature Extraction ##########################
# Using Resnet to extrect features from cropped images
# Load a pretrained ResNet/Densenet model

# Choose the model and whether to freeze layers
model_name_choice = 'vit'  # 'resnet', 'densenet', or 'vit'
freeze_layers_choice = False
model, device = load_model(model_name=model_name_choice, freeze_layers=freeze_layers_choice)


# Define a transformation to preprocess the ROIs
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Choose either 'original_images'or 'cropped_save_dir'
images_type = images_path(images='cropped_save_dir')

# Extract features from images
features_list = []
for image_file in os.listdir(images_type):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(images_type, image_file)
        features = extract_features(image_path, model, transform, device)
        if features is not None:
            features_list.append({'image_name': image_file, 'features': features})

# Convert features_list to a DataFrame
df = pd.DataFrame(features_list)

# Ensure valid_filenames_df only includes rows with matching image_name in df
valid_filenames_df = valid_filenames_df[valid_filenames_df['image_name'].isin(df['image_name'])]

# Print unmatched entries
unmatched = valid_filenames_df[~valid_filenames_df['image_name'].isin(df['image_name'])]

# Merge the DataFrames based on the 'image_name' column
merged_df = pd.merge(valid_filenames_df, df, on='image_name', how='inner')
#merged_df = merged_df.drop_duplicates(subset='image_name')

# # Check the structure and sample data of merged_df
# print("merged_df.shape:", merged_df.shape)
# print("merged_df sample:\n", merged_df.head())

# Split the features column into separate columns
features_df = pd.DataFrame(merged_df['features'].tolist(), index=merged_df.index)

# Check the structure of features_df
# print("features_df.shape:", features_df.shape)
# print("features_df sample:\n", features_df.head())

merged_df = merged_df.drop(columns=['features']).join(features_df)


# Compute Euclidean distances between feature vectors
feature_columns = list(features_df.columns)
distance_matrix = euclidean_distances(features_df[feature_columns])

# Verify Euclidean distances by checking a few sample distances
print("Sample distances:\n", distance_matrix[:5, :5])

# Find the 8 most similar images for each image
similar_images = []

for idx, row in merged_df.iterrows():
    distances = distance_matrix[idx]
    similar_indices = distances.argsort()[1:9]  # Exclude the first one (distance to itself)
    similar_image_ids = merged_df.iloc[similar_indices]['PID'].tolist()
    similar_image_names = merged_df.iloc[similar_indices]['image_name'].tolist()

      # Least similar images
    dissimilar_indices = distances.argsort()[-9:][::-1]  # Exclude the first one (distance to itself)
    dissimilar_image_ids = merged_df.iloc[dissimilar_indices]['PID'].tolist()
    dissimilar_image_names = merged_df.iloc[dissimilar_indices]['image_name'].tolist()

    similar_images.append({
        'image_name': row['image_name'],
        'image_id': row['PID'],
        'neighbors': row['neighbors'],
        'similar_image_ids': similar_image_ids,
        'dissimilar_image_ids': dissimilar_image_ids


    })

# Directly create the DataFrame from the list of dictionaries
similar_images_df = pd.DataFrame(similar_images)
similar_images_df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/similar_images_df.csv', index=False)


# Check the structure and sample data of similar_images_df
print("similar_images_df.shape:", similar_images_df.shape)
print("similar_images_df sample:\n", similar_images_df.head())


k = 8  # value for evauation metrics

#Similar images metrics
precision_list = []
recall_list = []
ndcg_list = []

for i, row in merged_df.iterrows():
    y_true = ensure_list_format(row['neighbors'])
    y_pred = ensure_list_format(similar_images_df.loc[similar_images_df['image_id'] == row['PID'], 'similar_image_ids'].values[0])

    # Print out the values to debug
    #print(f"y_true (ground truth): {y_true}")
   #print(f"y_pred (predictions): {y_pred}")

    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    ndcg = ndcg_at_k(y_true, y_pred, k)

    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)

# Create a DataFrame to store the results
evaluation_df = pd.DataFrame({
    'image_id': merged_df['PID'],
    'precision@k': precision_list,
    'recall@k': recall_list,
    'ndcg@k': ndcg_list
})

# Calculate the average metrics
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_ndcg = np.mean(ndcg_list)

print(f'Similar images average Precision@{k}: {average_precision:.4f}')
print(f'Similar images average Recall@{k}: {average_recall:.4f}')
print(f'Similar images average NDCG@{k}: {average_ndcg:.4f}')


#Dissimilar images metrics
dissimilar_precision_list = []
dissimilar_recall_list = []
dissimilar_ndcg_list = []

for i, row in merged_df.iterrows():
    diss_y_true = ensure_list_format(row['neighbors'])
    diss_y_pred = ensure_list_format(similar_images_df.loc[similar_images_df['image_id'] == row['PID'], 'dissimilar_image_ids'].values[0])

    # Print out the values to debug
    #print(f"y_true (ground truth): {y_true}")
   #print(f"y_pred (predictions): {y_pred}")

    precision = precision_at_k(diss_y_true, diss_y_pred, k)
    recall = recall_at_k(diss_y_true, diss_y_pred, k)
    ndcg = ndcg_at_k(diss_y_true, diss_y_pred, k)

    dissimilar_precision_list.append(precision)
    dissimilar_recall_list.append(recall)
    dissimilar_ndcg_list.append(ndcg)

# DataFrame to store the results
# evaluation_df = pd.DataFrame({
#     'image_id': merged_df['PID'],
#     'precision@k': dissimilar_precision_list,
#     'recall@k': dissimilar_recall_list,
#     'ndcg@k': dissimilar_ndcg_list
# })

# Calculate the average metrics
average_precision = np.mean(dissimilar_precision_list)
average_recall = np.mean(dissimilar_recall_list)
average_ndcg = np.mean(dissimilar_ndcg_list)

print(f'Dissimilar images average Precision@{k}: {average_precision:.4f}')
print(f'Dissimilar images average Recall@{k}: {average_recall:.4f}')
print(f'Dissimilar images average NDCG@{k}: {average_ndcg:.4f}')