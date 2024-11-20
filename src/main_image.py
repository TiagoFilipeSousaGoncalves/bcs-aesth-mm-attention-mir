# Imports
import argparse
import os
import numpy as np
import datetime
import random
import json
import shutil
import pandas as pd

# PyTorch Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

# Project Imports
from utilities_imgmodels import MODELS_DICT as models_dict
from utilities_preproc import sample_manager
from utilities_traintest import TripletDataset, train_model, eval_model

# WandB Imports
import wandb



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
    parser.add_argument('--config_json', type=str, required=False, default="config/config_image.json", help="The JSON configuration file.")
    parser.add_argument('--pickles_path', type=str, required=True, help="The path to the pickle files (to speed up training).")
    parser.add_argument('--results_path', type=str, required=False, help="The path to save the results.")
    parser.add_argument('--train_or_test', type=str, required=False, choices=["train", "test"], default="train", help="The execution setting: train or test.")
    parser.add_argument('--checkpoint_path', type=str, required=False, help="The path to the model checkpoints.")
    parser.add_argument('--verbose', action='store_true', default=False, help="Verbose.")
    args = parser.parse_args()


    # Build a configuration dictionary for WandB
    wandb_project_config = dict()
    
    # Create a timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get arguments
    gpu_id = args.gpu_id
    config_json_ = args.config_json
    pickles_path = args.pickles_path
    results_path = args.results_path
    train_or_test = args.train_or_test
    checkpoint_path = args.checkpoint_path
    verbose = args.verbose

    # If train
    if train_or_test == "train":
        experiment_results_path = os.path.join(results_path, timestamp)
        path_save = os.path.join(experiment_results_path, 'bin')

        for path in [experiment_results_path, pickles_path, path_save]:
            os.makedirs(path, exist_ok=True)

        # Open configuration JSON
        with open(config_json_, 'r') as j:
            config_json = json.load(j)

        # Copy configuration JSON to the experiment directory
        _ = shutil.copyfile(
            src=config_json_,
            dst=os.path.join(experiment_results_path, 'config.json')
        )
    
    else:
        path_save = os.path.join(checkpoint_path, 'bin')
        with open(os.path.join(checkpoint_path, 'config.json'), 'r') as j:
            config_json = json.load(j)



    # Set seed(s)
    set_seed(seed=config_json["seed"])

    # Create a device
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    if verbose:
        print(f"Using device: {device}")



    # Add information to WandB, if train
    if train_or_test == "train":
        wandb_project_config["seed"] = config_json["seed"]
        wandb_project_config["lr"] = config_json["lr"]
        wandb_project_config["num_epochs"] = config_json["num_epochs"]
        wandb_project_config[ "batch_size"] = config_json[ "batch_size"]
        wandb_project_config["margin"] = config_json["margin"]
        wandb_project_config["split_ratio"] = config_json["split_ratio"]
        wandb_project_config["catalogue_type"] = config_json["catalogue_type"]
        wandb_project_config["doctor_code"] = config_json["doctor_code"]
        wandb_project_config["model_name"] = config_json["model_name"]

        # Initialize WandB
        wandb_run = wandb.init(
            project="bcs-aesth-mm-attention-mir",
            name=config_json["model_name"]+'_'+timestamp,
            config=wandb_project_config
        )
        assert wandb_run is wandb.run



    # Preprocessing
    QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = sample_manager(pickles_path=pickles_path)



    if verbose:
        print("Summary of QNS:")
        for q in QNS_list_image_train:
            q.show_summary()
        for q in QNS_list_tabular_train:
            q.show_summary(str=False)



    # Create Model and Hyperparameters
    model_name = config_json["model_name"]
    model = models_dict[model_name]
    batch_size = config_json["batch_size"]
    margin = config_json["margin"]
    lr = config_json["lr"]
    num_epochs = config_json["num_epochs"]

    # Train Dataset & Dataloader
    train_dataset = TripletDataset(QNS_list=QNS_list_image_train, transform=model.get_transform())
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Test Dataset & Dataloader
    test_dataset = TripletDataset(QNS_list=QNS_list_image_test, transform=model.get_transform())
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss function and Optimizer
    criterion = TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    if train_or_test == "train":
        if verbose:
            print(f'Training {model_name}...')
        train_model(
            model,
            train_loader,
            test_loader,
            QNS_list_image_train,
            QNS_list_image_test,
            optimizer,
            criterion,
            num_epochs=num_epochs,
            device=device,
            path_save=path_save,
            wandb_run=wandb_run
        )
        wandb_run.finish()
    else:
        model.load_state_dict(torch.load(os.path.join(path_save, "model_final.pt"), map_location=device))
        train_acc, train_ndcg = eval_model(
            model=model,
            eval_loader=train_loader,
            QNS_list_eval=QNS_list_image_train,
            device=device
        )
        test_acc, test_ndcg = eval_model(
            model=model,
            eval_loader=test_loader,
            QNS_list_eval=QNS_list_image_test,
            device=device
        )
        results_dict = {
             "train_acc":[train_acc], 
             "train_ndcg":[train_ndcg],
            "test_acc":[test_acc],
            "test_ndcg":[test_ndcg]
        }
        eval_df = pd.DataFrame.from_dict(results_dict)
        if verbose:
            print(eval_df)
        eval_df.to_csv(os.path.join(checkpoint_path, "eval_results.csv"))