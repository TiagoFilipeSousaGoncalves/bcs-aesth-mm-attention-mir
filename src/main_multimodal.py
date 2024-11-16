# Imports
import argparse
import os
import random
import numpy as np
import datetime
import json
import shutil

# PyTorch Imports
import torch
import torch.optim as optim
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader

# Project Imports
from utilities_imgmodels import MODELS_DICT as models_img_dict
from utilities_preproc import sample_manager, QNS_structure
from utilities_tabmodels import collaborative_tabular_normalize
from utilities_tabmodels import MODELS_DICT as models_tab_dict
from utilities_traintest import TripletDataset, train_model

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
    parser = argparse.ArgumentParser(description='CINDERELLA BreLoAI Retrieval: Model Training, with multi-modal data.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--config_json', type=str, default="config/config_image.json", help="The JSON configuration file.")
    parser.add_argument('--images_resized_path', type=str, required=True, help="The path to the resized images.")
    parser.add_argument('--images_original_path', type=str, required=True, help="The path to the original images.")
    parser.add_argument('--csvs_path', type=str, required=True, help="The path to the CSVs with metadata.")
    parser.add_argument('--pickles_path', type=str, required=True, help="The path to the pickle files (to speed up training).")
    parser.add_argument('--img_model_weights_path', type=str, required=True, help="The path to the weights of the image model.")
    parser.add_argument('--results_path', type=str, required=True, help="The path to save the results.")
    parser.add_argument('--verbose', action='store_true', default=False, help="Verbose.")
    args = parser.parse_args()


    # Build a configuration dictionary for WandB
    wandb_project_config = dict()
    
    # Create a timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get arguments
    gpu_id = args.gpu_id
    config_json_ = args.config_json
    images_resized_path = args.images_resized_path
    images_original_path = args.images_original_path
    csvs_path = args.csvs_path
    pickles_path = args.pickles_path
    img_model_weights_path = args.img_model_weights_path
    results_path = args.results_path
    verbose = args.verbose

    # Build paths
    favorite_image_info = os.path.join(csvs_path, 'favorite_image_info.csv')
    patient_info = os.path.join(csvs_path, 'patient_info.csv')
    patient_images_info = os.path.join(csvs_path, 'patient_images.csv')
    catalogue_info = os.path.join(csvs_path, 'catalogue_info.csv')
    catalogue_user_info = os.path.join(csvs_path, 'catalogue_user_info.csv')
    experiment_results_path = os.path.join(results_path, timestamp)
    path_save = os.path.join(experiment_results_path, 'bin')


    # Create results path (if needed)
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

    # Set seed(s)
    set_seed(seed=config_json["seed"])

    # Create a device
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    if verbose:
        print(f"Using device: {device}")



    # Add information to WandB
    wandb_project_config["seed"] = config_json["seed"]
    wandb_project_config["lr"] = config_json["lr"]
    wandb_project_config["num_epochs"] = config_json["num_epochs"]
    wandb_project_config[ "batch_size"] = config_json[ "batch_size"]
    wandb_project_config["margin"] = config_json["margin"]
    wandb_project_config["split_ratio"] = config_json["split_ratio"]
    wandb_project_config["catalogue_type"] = config_json["catalogue_type"]
    wandb_project_config["doctor_code"] = config_json["doctor_code"]
    wandb_project_config["model_img_name"] = config_json["model_img_name"]
    wandb_project_config["model_tab_name"] = config_json["model_tab_name"]

    # Initialize WandB
    wandb_run = wandb.init(
        project="bcs-aesth-mm-attention-mir",
        name=config_json["model_img_name"]+'_'+config_json["model_tab_name"]+'_'+timestamp,
        config=wandb_project_config
    )
    assert wandb_run is wandb.run


    # TODO: Delete after testing    
    # Required Paths
    # current_directory = os.getcwd()
    # images_path='/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz-v2/'
    # csvs_path =current_directory+'/csvs/'
    # favorite_image_info = csvs_path + 'favorite_image_info.csv'
    # patient_info = csvs_path + 'patient_info.csv'
    # patient_images_info = csvs_path + 'patient_images.csv'
    # catalogue_info = csvs_path + 'catalogue_info.csv'
    # catalogue_user_info = csvs_path + 'catalogue_user_info.csv'
    # train_pickle_path = current_directory + '/pickles_clean/qns_list_train_img_2.pkl'
    # test_pickle_path  = current_directory + '/pickles_clean/qns_list_test_img_2.pkl'
    # train_pickle_path_tab = current_directory + '/pickles_clean/qns_list_train_tab_2.pkl'
    # test_pickle_path_tab  = current_directory + '/pickles_clean/qns_list_test_tab_2.pkl'
    # path_save = current_directory + '/bin_clean/'

    # split_ratio=0.8
    # catalogue_type = 'E'
    # doctor_code=-1 # 39 57 36 -1

    # Read dataset
    QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = sample_manager(pickles_path=pickles_path)



    # Create model, load model weights and activate evaluation mode
    model_img = models_img_dict[config_json["model_img_name"]]
    model_img.load_state_dict(torch.load(img_model_weights_path, map_location=device))
    model_img.to(device)
    model_img.eval()
    transform = model_img.get_transform()



    # Save query image outputs - Train
    train_outs_query = []
    for qns in QNS_list_image_train:
            query = qns.query_vector
            query_input = transform(query)

            # Ensure the query input is a tensor and has the correct shape
            if isinstance(query_input, np.ndarray):
                query_input = torch.tensor(query_input)
            
            if len(query_input.shape) == 3:  # Add batch dimension if missing
                query_input = query_input.unsqueeze(0)
            
            query_input = query_input.to(device)
            query_out = model_img(query_input)
            query_out = query_out.cpu().detach().numpy()
            # query_out = query_out.detach().numpy()
            train_outs_query.append(query_out[0])

    # Save neighbour image outputs - Train
    train_outs_rtr = []
    for qns in QNS_list_image_train:
            aux = []
            rtr_vectors = qns.neighbor_vectors
            for z in range(len(rtr_vectors)):
                rtr_input = transform(rtr_vectors[z])
                
                # Ensure the query input is a tensor and has the correct shape
                if isinstance(rtr_input, np.ndarray):
                    rtr_input = torch.tensor(rtr_input)
                
                # Add batch dimension if missing
                if len(rtr_input.shape) == 3:
                    rtr_input = rtr_input.unsqueeze(0)
                
                rtr_input = rtr_input.to(device)
                rtr_input = model_img(rtr_input)
                rtr_input = rtr_input.cpu().detach().numpy()
                # rtr_input = rtr_input.detach().numpy()

                aux.append(rtr_input[0])
            train_outs_rtr.append(aux)


    # Save query image outputs - Test
    test_outs_query = []
    for qns in QNS_list_image_test:
            query = qns.query_vector
            transform = model_img.get_transform()
            query_input = transform(query)
            if isinstance(query_input, np.ndarray):
                query_input = torch.tensor(query_input)
            
            # Add batch dimension if missing
            if len(query_input.shape) == 3:
                query_input = query_input.unsqueeze(0)
            
            query_input = query_input.to(device)
            query_out = model_img(query_input)
            query_out = query_out.cpu().detach().numpy()
            # query_out = query_out.detach().numpy()
            test_outs_query.append(query_out[0])

    # Save neighbour image outputs - Test
    test_outs_rtr = []
    for qns in QNS_list_image_test:
            aux =[]
            rtr_vectors = qns.neighbor_vectors
            for z in range(len(rtr_vectors)):
                rtr_input = transform(rtr_vectors[z])
                # Ensure the query input is a tensor and has the correct shape
                if isinstance(rtr_input, np.ndarray):
                    rtr_input = torch.tensor(rtr_input)
                
                # Add batch dimension if missing
                if len(rtr_input.shape) == 3:
                    rtr_input = rtr_input.unsqueeze(0)
                
                rtr_input = rtr_input.to(device)
                rtr_input = model_img(rtr_input)
                rtr_input = rtr_input.cpu().detach().numpy()
                # rtr_input = rtr_input.detach().numpy()

                aux.append(rtr_input[0])
            test_outs_rtr.append(aux)



    # TODO: Erase after testing
    # with open(train_pickle_path_tab, 'rb') as file:
    #             QNS_list_train_t = pickle.load(file)
    # with open(test_pickle_path_tab, 'rb') as file:
    #             QNS_list_test_t = pickle.load(file)



    # Create a new QNS for MultiModal
    # Train
    QNS_list_train_tab = []
    count = 0
    for qns in QNS_list_tabular_train: 
        qns_element = QNS_structure()
        itm = qns.query_vector
        # id = qns.query_vector_id
        itm = np.append(itm, train_outs_query[count])
        qns_element.set_query_vector(itm, qns.query_vector_id)

        for jdx in range(len(qns.neighbor_vectors_id)): 
            itm = qns.neighbor_vectors[jdx]
            # id = qns.neighbor_vectors_id[jdx]
            itm = np.append(itm,train_outs_rtr[count][jdx])
            qns_element.add_neighbor_vector(itm, qns.neighbor_vectors_id[jdx])
        qns_element.calculate_expert_score()
        QNS_list_train_tab.append(qns_element)
        count += 1

    # Test
    QNS_list_test_tab = []
    count = 0
    for qns in QNS_list_tabular_test:
        qns_element = QNS_structure()
        itm = qns.query_vector
        # id = qns.query_vector_id
        itm = np.append(itm,test_outs_query[count])
        qns_element.set_query_vector(itm, qns.query_vector_id)

        for jdx in range(len(qns.neighbor_vectors_id)): 
            itm = qns.neighbor_vectors[jdx]
            # id = qns.neighbor_vectors_id[jdx]
            itm = np.append(itm,test_outs_rtr[count][jdx])
            qns_element.add_neighbor_vector(itm, qns.neighbor_vectors_id[jdx])
        qns_element.calculate_expert_score()
        QNS_list_test_tab.append(qns_element)
        count += 1


    # print('Starting...')
    # Configs
    # np.random.seed(10)
    # torch.manual_seed(10)

    # lr=0.0001
    # num_epochs=200
    # batch_size=512
    # margin = 0.0001
    # split_ratio=0.8
    # catalogue_type = 'E'
    # doctor_code=-1 # 39 57 36 -1


    # Use Train QNS to obtain the Min/Max values for collaborative tabular normalization
    min_max_values = collaborative_tabular_normalize(QNS_list_train_tab)

    # Apply these Min/Max values on Test for collaborative tabular normalization
    min_max_values_ = collaborative_tabular_normalize(QNS_list_test_tab, min_max_values)

    assert min_max_values == min_max_values_, f"min_max_values {min_max_values} and  min_max_values_ {min_max_values_} should be the same."



    # TODO: WIP
    # path_save = current_directory + '/bin_MultiModal_v2/'


    # Get Tabular model
    # models = {
    #     "TabularMLP_773_300_20": TabularMLP(773, 200, 20 )
    # }
    model_tab = models_tab_dict[config_json["model_tab_name"]]

    # Define Dataset & Dataloaders & Optimization Parameters
    train_dataset = TripletDataset(QNS_list=QNS_list_train_tab, transform=model_tab.get_transform())
    test_dataset = TripletDataset(QNS_list=QNS_list_test_tab, transform=model_tab.get_transform())
    train_loader = DataLoader(train_dataset, batch_size=config_json["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config_json["batch_size"], shuffle=False)
    criterion = TripletMarginLoss(margin=config_json["margin"], p=2)
    optimizer = optim.Adam(model_tab.parameters(), lr=config_json["lr"])

    if verbose:
        print(
             'Training Multi-modal with ', 
             config_json["model_img_name"], 
             ' (image) and ', 
             config_json["model_tab_name"], 
             ' (tabular).'
        )

    train_model(
         model=model_tab,
         train_loader=train_loader, 
         test_loader=test_loader, 
         QNS_list_train=QNS_list_train_tab, 
         QNS_list_test=QNS_list_test_tab, 
         optimizer=optimizer, 
         criterion=criterion, 
         num_epochs=config_json["num_epochs"], 
         device=device, 
         path_save=path_save,
         wandb_run=wandb_run  
    )
    wandb_run.finish()
