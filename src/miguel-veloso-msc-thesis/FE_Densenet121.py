import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torchvision.transforms
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics.pairwise import euclidean_distances

import cv2

# Import the get_filenames function from utilities_file
from Utilities_file import (
    get_filenames, 
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
    ensure_list_format
)
from Models import UNet


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
csv_file_1 = base_path + 'df_combined.csv'

# Load the DataFrame with image and mask file names
valid_filenames_df = pd.read_csv(csv_file_1)

# Define the directory path to be removed
directory_path = "/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/"

# Apply the extract_name function to the PID column
valid_filenames_df['image_name'] = valid_filenames_df['image PID'].apply(lambda x: extract_name(x, directory_path))

# Save the DataFrame to a CSV file if needed
#valid_filenames_df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/check_Validnames.csv', index=False)

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

                            # Save the cropped image
                            #cropped_img_save_path = os.path.join(cropped_save_dir, f'{img_filenames[j]}')
                           # cv2.imwrite(cropped_img_save_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                        # Save the image with bounding boxes
                       # image_with_boxes_save_path = os.path.join(save_dir, f'boxed_{img_filenames[j]}')
                       # cv2.imwrite(image_with_boxes_save_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

        # Calculate and print the average test MSE and Dice scores
        avg_test_dice = np.mean(test_dice_scores)
        avg_test_mse = np.mean(test_mse_scores)
        print(f"Average Test Dice Score: {avg_test_dice:.4f}")
        print(f"Average Test MSE: {avg_test_mse:.4f}")

        return avg_test_dice, avg_test_mse

# Initialize the model
model = UNet(dropout_rate=0.2).to(device)
# Define a loss function
mse_loss = torch.nn.MSELoss()

# Load the best saved model before testing
model.load_state_dict(torch.load('unet_model.pth'))
test_dice, test_loss = test_model(model, test_loader, bce_dice_loss, device)

cropped_save_dir='/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/Cropped_images/'

####################### Densenet Feature Extraction ##########################
#densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1')
densenet.eval()
densenet = nn.Sequential(*list(densenet.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
densenet = densenet.to(device)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Extract features from images
features_list = []
for image_file in os.listdir(cropped_save_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(cropped_save_dir, image_file)
        features = extract_features(image_path, densenet, preprocess, device)
        if features is not None:
            features_list.append({'image_name': image_file, 'features': features})


# Convert features_list to a DataFrame
df = pd.DataFrame(features_list)

###just a quick method to create pipeline, it should be fixed after
# Filter valid_filenames_df to only include rows with matching image_name in df
valid_filenames_df = valid_filenames_df[valid_filenames_df['image_name'].isin(df['image_name'])]
#valid_filenames_df

#df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/df_features.csv', index=False)

# Merge the DataFrames based on the 'image_name' column
merged_df = pd.merge(valid_filenames_df, df, on='image_name', how='inner')

# Split the features column into separate columns
features_df = pd.DataFrame(merged_df['features'].tolist(), index=merged_df.index)
merged_df = merged_df.drop(columns=['features']).join(features_df)
# # Save the DataFrame to a CSV file if needed
#merged_df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/features_with_ids.csv', index=False)

# Compute Euclidean distances between feature vectors
feature_columns = list(features_df.columns)
distance_matrix = euclidean_distances(features_df[feature_columns])

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

# Save the DataFrame to a CSV file
#similar_images_df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/rankings.csv', index=False)

#Evaluation metrics

k = 8  # Set the value of k for Precision@k, Recall@k, and NDCG@k

precision_list = []
recall_list = []
ndcg_list = []

for i, row in merged_df.iterrows():
    y_true = ensure_list_format(row['neighbors'])
    y_pred = ensure_list_format(similar_images_df.loc[similar_images_df['image_id'] == row['PID'], 'similar_image_ids'].values[0])

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

# Save the evaluation DataFrame to a CSV file
#evaluation_df.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/densenet_evaluation_metrics.csv', index=False)