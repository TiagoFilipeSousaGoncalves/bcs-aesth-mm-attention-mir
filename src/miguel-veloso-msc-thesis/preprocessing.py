import numpy as np
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss
import pandas as pd

from PreprocessingUtilities import sample_manager

base_path = '/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/'
#path = /nas-ctm01/homes/cmveloso/miguel-veloso-msc-thesis/Dataset/Excellent_Good/
#Required Paths
current_directory = os.getcwd()
print(current_directory)
images_path='/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'
csvs_path ='/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs/'
favorite_image_info = csvs_path + 'favorite_image_info.csv'
patient_info = csvs_path + 'patient_info.csv'
patient_images_info = csvs_path + 'patient_images.csv'
catalogue_info = csvs_path + 'catalogue_info.csv'
catalogue_user_info = csvs_path + 'catalogue_user_info.csv'
pickle_path = current_directory + '/Dataset/Fair_Poor/'
#path_save = '../bin/'

# Configs
np.random.seed(10)
torch.manual_seed(10)
device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
lr=0.00001
num_epochs=1
batch_size=16
margin = 0.0001
split_ratio=0.8
catalogue_type = 'P'
doctor_code=-1 # 39 57 36 -1

# # Preprocessing
# QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = \
# sample_manager(images_path, pickle_path, catalogue_info, catalogue_user_info, 
# patient_info, favorite_image_info, patient_images_info, catalogue_type=catalogue_type, doctor_code=doctor_code, split_ratio=split_ratio, default=False)

# for q in QNS_list_image_train:
#      q.show_summary()
# for q in QNS_list_tabular_train:
#      q.show_summary(str=False)

# # Down-Sampeling
# QNS_list_image_train = QNS_list_image_train[0:2]
# QNS_list_image_test = QNS_list_image_test[0:2]

df_train, df_test = sample_manager(images_path, pickle_path, catalogue_info, catalogue_user_info, patient_info, favorite_image_info, patient_images_info, catalogue_type='E', doctor_code=-1, split_ratio=0.8, default=True)

df_combined = pd.concat([df_train, df_test])

# Save the combined DataFrame to a CSV file
df_combined.to_csv('/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/df_combined_p.csv', index=False)

print("Combined DataFrame saved successfully.")

