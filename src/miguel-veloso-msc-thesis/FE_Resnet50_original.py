import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

import torch.nn as nn
import torchvision.models as models

from sklearn.metrics.pairwise import euclidean_distances

from Utilities_file import (
    get_filenames, 
    extract_features,
    extract_name, 
    precision_at_k, 
    ndcg_at_k, 
    recall_at_k, 
    dcg_at_k, 
    unnormalize_tensor, 
    save_image_from_array,
    ensure_list_format
)
from Models import UNet

# Path to the CSV file
base_path = '/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/'
csv_file_1 = base_path + 'df_combined.csv'

# Load the DataFrame with image and mask file names
valid_filenames_df = pd.read_csv(csv_file_1)

# Define the directory path to be removed
directory_path = "/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/"

# Apply the extract_name function to the PID column
valid_filenames_df['image_name'] = valid_filenames_df['image PID'].apply(lambda x: extract_name(x, directory_path))


# Load a pretrained ResNet model
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.eval()
resnet = nn.Sequential(*list(resnet.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Define a transformation to preprocess the ROIs
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features
def extract_features(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

# Directory containing cropped images
original_images = '/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'

valid_filenames_df = valid_filenames_df.drop_duplicates(subset='image_name')

# Extract features from images listed in valid_filenames_df
features_list = []
for image_file in valid_filenames_df['image_name']:
    image_path = os.path.join(original_images, image_file)
    if os.path.isfile(image_path) and (image_file.endswith('.jpg') or image_file.endswith('.png')):
        features = extract_features(image_path, resnet, transform, device)
        if features is not None:
            features_list.append({'image_name': image_file, 'features': features})


# Convert features_list to a DataFrame
df = pd.DataFrame(features_list)

# Ensure valid_filenames_df only includes rows with matching image_name in df
valid_filenames_df = valid_filenames_df[valid_filenames_df['image_name'].isin(df['image_name'])]


# Merge the DataFrames based on the 'image_name' column
merged_df = pd.merge(valid_filenames_df, df, on='image_name', how='inner')

# Check the structure and sample data of merged_df
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

    similar_images.append({
        'image_name': row['image_name'],
        'image_id': row['PID'],
        'neighbors': row['neighbors'],
        'similar_image_ids': similar_image_ids
    })

# Directly create the DataFrame from the list of dictionaries
similar_images_df = pd.DataFrame(similar_images)

# Check the structure and sample data of similar_images_df
print("similar_images_df.shape:", similar_images_df.shape)
print("similar_images_df sample:\n", similar_images_df.head())


k = 8  # value for evauation metrics

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

print(f'Average Precision@{k}: {average_precision:.4f}')
print(f'Average Recall@{k}: {average_recall:.4f}')
print(f'Average NDCG@{k}: {average_ndcg:.4f}')