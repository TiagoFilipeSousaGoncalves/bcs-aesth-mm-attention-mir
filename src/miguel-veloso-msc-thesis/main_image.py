import numpy as np
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

from ImageUtilities import DeiT_Base_Patch16_224, Beit_Base_Patch16_224, ResNet50_Base_224
from TrainUtilities import TripletDataset, train_triplets, save_model
from PreprocessingUtilities import sample_manager

#Required Paths
base_path = '/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/csv/'
path = '/nas-ctm01/homes/cmveloso/miguel-veloso-msc-thesis/Dataset/Excellent_Good/'
#Required Paths
current_directory = os.getcwd()
print(current_directory)
images_path='/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz/'
#images_path='/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/Cropped_images/'
csvs_path ='/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs/'
favorite_image_info = csvs_path + 'favorite_image_info.csv'
patient_info = csvs_path + 'patient_info.csv'
patient_images_info = csvs_path + 'patient_images.csv'
catalogue_info = csvs_path + 'catalogue_info.csv'
catalogue_user_info = csvs_path + 'catalogue_user_info.csv'
pickle_path = current_directory + '/Dataset/Excellent_Good/'
path_save = '/nas-ctm01/datasets/private/CINDERELLA/Pred_masks/'

# Configs
np.random.seed(10)
torch.manual_seed(10)
device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
lr=0.00001
num_epochs=5
batch_size=16
margin = 0.0001
split_ratio=0.8
catalogue_type = 'E'
doctor_code=-1 # 39 57 36 -1

# Preprocessing
QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = \
sample_manager(images_path, pickle_path, catalogue_info, catalogue_user_info, 
patient_info, favorite_image_info, patient_images_info, catalogue_type=catalogue_type, doctor_code=doctor_code, split_ratio=split_ratio, default=False)

# for q in QNS_list_image_train:
#     q.show_summary()
# for q in QNS_list_tabular_train:
#     q.show_summary(str=False)

# # Down-Sampeling
# QNS_list_image_train = QNS_list_image_train[0:2]
# QNS_list_image_test = QNS_list_image_test[0:2]

# Implemented Model
models = {
    "DeiT_Base_Patch16_224": DeiT_Base_Patch16_224(),
    "Beit_Base_Patch16_224": Beit_Base_Patch16_224(),
    "ResNet50_Base_224": ResNet50_Base_224()
  
}

for model_name, model in models.items():
    
    # Define Dataset & Dataloaders & Optimization Parameters
    train_dataset = TripletDataset(images_path, QNS_list_image_train, transform=model.get_transform())
    test_dataset  = TripletDataset(images_path, QNS_list_image_test,  transform=model.get_transform())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # later it should bea turned on ...
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    criterion     = TripletMarginLoss(margin=margin, p=2)
    optimizer     = optim.Adam(model.parameters(), lr=lr)

    print(f'Training {model_name}...')
    model, epoch_loss, epoch = train_triplets(model, train_loader, test_loader, QNS_list_image_train, QNS_list_image_test, optimizer, criterion, num_epochs=num_epochs, model_name=model_name, device=device, path_save=path_save)
    
    print(f'Saving {model_name}...')
    save_model(model, f'{path_save}{model_name}/Finale.pl')
    print(f'Done {model_name}!')