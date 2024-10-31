# Imports
import argparse
import os
import numpy as np
import datetime
import random
import json
import shutil

# PyTorch Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

# Project Imports
from utilities_imgmodels import MODELS_DICT as models_dict
from utilities_preproc import sample_manager
from utilities_traintest import TripletDataset, train_triplets, save_model


# Function: See the seed for reproducibility purposes
def set_seed(seed=10):

    # Random Seed
    random.seed(seed)

    # Environment Variable Seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy Seed
    np.random.seed(seed)

    # PyTorch Seed(s)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return



if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='CINDERELLA BreLoAI Retrieval: Model Training, with image data.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--config_json', type=str, default="config/config_image.json", help="The JSON configuration file.")
    parser.add_argument('--images_path', type=str, required=True, help="The path to the images.")
    parser.add_argument('--csvs_path', type=str, required=True, help="The path to the CSVs with metadata.")
    parser.add_argument('--results_path', type=str, required=True, help="The path to save the results.")
    parser.add_argument('--verbose', action='store_true', default=False, help="Verbose.")
    args = parser.parse_args()


    # TODO: Erase after testing
    # Required Paths
    # current_directory = os.getcwd()
    # images_path='../data/images/'
    # csvs_path ='../data/csvs/'
    
    # Create a timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get arguments
    gpu_id = args.gpu_id
    config_json_ = args.config_json
    images_path = args.images_path
    csvs_path = args.csvs_path
    results_path = args.results_path
    verbose = args.verbose

    # Build paths
    favorite_image_info = os.path.join(csvs_path, 'favorite_image_info.csv')
    patient_info = os.path.join(csvs_path, 'patient_info.csv')
    patient_images_info = os.path.join(csvs_path, 'patient_images.csv')
    catalogue_info = os.path.join(csvs_path, 'catalogue_info.csv')
    catalogue_user_info = os.path.join(csvs_path, 'catalogue_user_info.csv')
    experiment_results_path = os.path.join(results_path, timestamp)
    pickle_path = os.path.join(experiment_results_path, 'data', 'pickles')
    path_save = os.path.join(experiment_results_path, 'bin')


    # Create results path (if needed)
    for path in [experiment_results_path, pickle_path, path_save]:
        os.makedirs(path)


    # Open configuration JSON
    with open(config_json_, 'r') as j:
        config_json = json.load(j)

    # Copy configuration JSON to the experiment directory
    _ = shutil.copyfile(
        src=config_json_,
        dst=os.path.join(experiment_results_path, 'config.json')
    )

    # Set seed(s)
    set_seed(seed=config_json["seed"])

    # Create a device
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    if verbose:
        print(f"Using device: {device}")

    # Configs
    # np.random.seed(10)
    # torch.manual_seed(10)
    # device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"
    # print(f"Using device: {device}")
    # lr=0.00001
    # num_epochs=1
    # batch_size=16
    # margin = 0.0001
    # split_ratio=0.8
    # catalogue_type = 'E'
    # doctor_code=-1 # 39 57 36 -1

    # Preprocessing
    QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = sample_manager(
        samples_path=images_path,
        pickle_path=pickle_path,
        catalogue_info=catalogue_info,
        catalogue_user_info=catalogue_user_info,
        patient_info=patient_info,
        favorite_image_info=favorite_image_info,
        patient_images_info=patient_images_info,
        catalogue_type=config_json["catalogue_type"],
        doctor_code=config_json["doctor_code"],
        split_ratio=config_json["split_ratio"],
        default=False
    )

    # for q in QNS_list_image_train:
    #     q.show_summary()
    # for q in QNS_list_tabular_train:
    #     q.show_summary(str=False)

    # # Down-Sampeling
    # QNS_list_image_train = QNS_list_image_train[0:2]
    # QNS_list_image_test = QNS_list_image_test[0:2]

    # Create Model and Hyperparameters
    model_name = config_json["model_name"]
    model = models_dict[model_name]
    batch_size = config_json["batch_size"]
    margin = config_json["margin"]
    lr = config_json["lr"]
    num_epochs = config_json["num_epochs"]

    # Train Dataset & Dataloader
    train_dataset = TripletDataset(images_path, QNS_list_image_train, transform=model.get_transform())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # later it should bea turned on ...

    # Test Dataset & Dataloader
    test_dataset = TripletDataset(images_path, QNS_list_image_test,  transform=model.get_transform())
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    # Loss function and Optimizer
    criterion = TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    if verbose:
        print(f'Training {model_name}...')
    model, _, _ = train_triplets(
        model,
        train_loader,
        test_loader,
        QNS_list_image_train,
        QNS_list_image_test,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        model_name=model_name,
        device=device,
        path_save=path_save
    )
    
    # Save model
    if verbose:
        print(f'Saving {model_name}...')
    save_model(model, os.path.join(path_save, "model_final.pt"))
