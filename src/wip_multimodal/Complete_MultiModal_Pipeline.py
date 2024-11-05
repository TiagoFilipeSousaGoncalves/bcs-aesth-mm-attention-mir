import numpy as np
import torch
import pickle
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

from ImageUtilities import Google_Base_Patch16_224, DeiT_Base_Patch16_224, Beit_Base_Patch16_224, DinoV2_Base_Patch16_224, ResNet50_Base_224, VGG16_Base_224
from ImageUtilities import Google_Base_Patch16_224_MLP, DeiT_Base_Patch16_224_MLP, Beit_Base_Patch16_224_MLP, DinoV2_Base_Patch16_224_MLP, ResNet50_Base_224_MLP, VGG16_Base_224_MLP

from TrainTestUtilities import TripletDataset, train_triplets, save_model
from PreprocessingUtilities import Sample_Manager
from TrainTestUtilities import TripletDataset, train_triplets, save_model
from PreprocessingUtilities import Sample_Manager,get_query_neighbor_elements_path,QNS_structure,edit_name_incase_using_resized
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

from wip_multimodal.TabularUtilities import TabularMLP
from TrainTestUtilities import TripletDataset, train_triplets, save_model, evaluate_nddg
from PreprocessingUtilities import Sample_Manager

def collaborative_tabular_normalize(qns_list, min_max_values=None):
    if min_max_values is not None:
        vec_len = len(min_max_values)
    else:
        vec_len = len(qns_list[0].query_vector)  # Assuming all vectors have the same length
        min_max_values = []

    all_elements = [[] for _ in range(vec_len)]

    # Collecting all elements for each position from both query and neighbor vectors
    for qns in qns_list:
        for i in range(vec_len):
            all_elements[i].append(qns.query_vector[i])
            for neighbor_vector in qns.neighbor_vectors:
                all_elements[i].append(neighbor_vector[i])
    
    # If min_max_values is provided, use it for normalization
    if min_max_values:
        for i in range(vec_len):
            min_val, max_val = min_max_values[i]
            all_elements[i] = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in all_elements[i]]
    else:
        # Normalizing each position across all instances and storing min-max values
        for i in range(vec_len):
            min_val = np.min(all_elements[i])
            max_val = np.max(all_elements[i])
            all_elements[i]  = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in all_elements[i]]
            min_max_values.append((min_val, max_val))
        print("Min_Max Values are: ", min_max_values)

    # Updating the vectors in QNS_structure instances
    for qns in qns_list:
        for i in range(vec_len):
            qns.query_vector[i] = all_elements[i].pop(0)
            for neighbor_vector in qns.neighbor_vectors:
                neighbor_vector[i] = all_elements[i].pop(0)

    return min_max_values



device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_class, filepath, device='cpu'):

    model = model_class  # Create an instance of the model
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath} onto {device}")
    return model

#Required Paths
current_directory = os.getcwd()
images_path='/nas-ctm01/datasets/private/CINDERELLA/breloai-rsz-v2/'
csvs_path =current_directory+'/csvs/'
favorite_image_info = csvs_path + 'favorite_image_info.csv'
patient_info = csvs_path + 'patient_info.csv'
patient_images_info = csvs_path + 'patient_images.csv'
catalogue_info = csvs_path + 'catalogue_info.csv'
catalogue_user_info = csvs_path + 'catalogue_user_info.csv'
train_pickle_path = current_directory + '/pickles_clean/qns_list_train_img_2.pkl'
test_pickle_path  = current_directory + '/pickles_clean/qns_list_test_img_2.pkl'
train_pickle_path_tab = current_directory + '/pickles_clean/qns_list_train_tab_2.pkl'
test_pickle_path_tab  = current_directory + '/pickles_clean/qns_list_test_tab_2.pkl'
path_save = current_directory + '/bin_clean/'

split_ratio=0.8
catalogue_type = 'E'
doctor_code=-1 # 39 57 36 -1


# Read Dataset
QNS_list_train, QNS_list_test = Sample_Manager(images_path, train_pickle_path, test_pickle_path, catalogue_info, catalogue_user_info, 
patient_info, favorite_image_info, patient_images_info, catalogue_type=catalogue_type, doctor_code=doctor_code, split_ratio=split_ratio, default=False)

#Define Model and Load Trained Image Model
model_name = "DinoV2_Base_Patch16_224"
model_type = DinoV2_Base_Patch16_224()
model = load_model(model_type,f'{path_save}{model_name}/Finale.pl')
transform=model.get_transform()
model = model.to(device)

#Save query image outputs - Train
train_outs_query = []
for qns in (QNS_list_train):
        query = qns.query_vector
        query_input = transform(query)

        # Ensure the query input is a tensor and has the correct shape
        if isinstance(query_input, np.ndarray):
            query_input = torch.tensor(query_input)
        
        if len(query_input.shape) == 3:  # Add batch dimension if missing
            query_input = query_input.unsqueeze(0)
        
        query_input = query_input.to(device)
        query_out = model(query_input)
        query_out = query_out.to('cpu')
        query_out = query_out.detach().numpy()
        train_outs_query.append(query_out[0])

#Save neighbour image outputs - Train
train_outs_rtr = []
for qns in (QNS_list_train):
        aux =[]
        rtr_vectors = qns.neighbor_vectors
        for z in range(len(rtr_vectors)):
            rtr_input = transform(rtr_vectors[z])
            # Ensure the query input is a tensor and has the correct shape
            if isinstance(rtr_input, np.ndarray):
                rtr_input = torch.tensor(rtr_input)
            
            if len(rtr_input.shape) == 3:  # Add batch dimension if missing
                rtr_input = rtr_input.unsqueeze(0)
            
            rtr_input = rtr_input.to(device)
            rtr_input = model(rtr_input)
            rtr_input = rtr_input.to('cpu')
            rtr_input = rtr_input.detach().numpy()

            aux.append(rtr_input[0])
        train_outs_rtr.append(aux)

#Save query image outputs - Test
test_outs_query = []
for qns in (QNS_list_test):
        query = qns.query_vector
        transform=model.get_transform()
        query_input = transform(query)
        if isinstance(query_input, np.ndarray):
            query_input = torch.tensor(query_input)
        
        if len(query_input.shape) == 3:  # Add batch dimension if missing
            query_input = query_input.unsqueeze(0)
        
        query_input = query_input.to(device)
        query_out = model(query_input)
        query_out = query_out.to('cpu')
        query_out = query_out.detach().numpy()
        test_outs_query.append(query_out[0])



#Save neighbour image outputs - Test
test_outs_rtr = []
for qns in (QNS_list_test):
        aux =[]
        rtr_vectors = qns.neighbor_vectors
        for z in range(len(rtr_vectors)):
            rtr_input = transform(rtr_vectors[z])
            # Ensure the query input is a tensor and has the correct shape
            if isinstance(rtr_input, np.ndarray):
                rtr_input = torch.tensor(rtr_input)
            
            if len(rtr_input.shape) == 3:  # Add batch dimension if missing
                rtr_input = rtr_input.unsqueeze(0)
            
            rtr_input = rtr_input.to(device)
            rtr_input = model(rtr_input)
            rtr_input = rtr_input.to('cpu')
            rtr_input = rtr_input.detach().numpy()

            aux.append(rtr_input[0])
        test_outs_rtr.append(aux)

with open(train_pickle_path_tab, 'rb') as file:
            QNS_list_train_t = pickle.load(file)
with open(test_pickle_path_tab, 'rb') as file:
            QNS_list_test_t = pickle.load(file)



#Create a New QNS for MultiModal
QNS_list_train_tab = []
QNS_list_test_tab = []
count=0
for qns in (QNS_list_train_t): 
    qns_element = QNS_structure()
    itm = qns.query_vector
    id = qns.query_vector_id
    itm = np.append(itm,train_outs_query[count])
    qns_element.set_query_vector(itm,  qns.query_vector_id)

    for jdx in range(len(qns.neighbor_vectors_id)): 
        itm = qns.neighbor_vectors[jdx]
        id = qns.neighbor_vectors_id[jdx]
        itm = np.append(itm,train_outs_rtr[count][jdx])
        qns_element.add_neighbor_vector(itm, qns.neighbor_vectors_id[jdx])
    qns_element.calculate_expert_score()
    QNS_list_train_tab.append(qns_element)
    count +=1
count=0
for qns in (QNS_list_test_t): 
    qns_element = QNS_structure()
    itm = qns.query_vector
    id = qns.query_vector_id
    itm = np.append(itm,test_outs_query[count])
    qns_element.set_query_vector(itm,  qns.query_vector_id)

    for jdx in range(len(qns.neighbor_vectors_id)): 
        itm = qns.neighbor_vectors[jdx]
        id = qns.neighbor_vectors_id[jdx]
        itm = np.append(itm,test_outs_rtr[count][jdx])
        qns_element.add_neighbor_vector(itm, qns.neighbor_vectors_id[jdx])
    qns_element.calculate_expert_score()
    QNS_list_test_tab.append(qns_element)
    count +=1


print('Starting...')
# Configs
np.random.seed(10)
torch.manual_seed(10)
device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
lr=0.0001
num_epochs=200
batch_size=512
margin = 0.0001
split_ratio=0.8
catalogue_type = 'E'
doctor_code=-1 # 39 57 36 -1

min_max_values = collaborative_tabular_normalize(QNS_list_train_tab)
collaborative_tabular_normalize(QNS_list_test_tab, min_max_values)

path_save = current_directory + '/bin_MultiModal_v2/'


# Implemented Model
models = {
    "Tabular-MLP": TabularMLP(773, 200, 20 )
}

for model_name, model in models.items():
    # # Define Dataset & Dataloaders & Optimization Parameters
    train_dataset = TripletDataset('', QNS_list_train_tab, transform=model.get_transform())
    test_dataset  = TripletDataset('', QNS_list_test_tab,  transform=model.get_transform())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # later it should bea turned on ...
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    criterion     = TripletMarginLoss(margin=margin, p=2)
    optimizer     = optim.Adam(model.parameters(), lr=lr)

    print(f'Training {model_name}...')
    model, _, _ = train_triplets(model, train_loader, test_loader, QNS_list_train_tab, QNS_list_test_tab, optimizer, criterion, num_epochs=num_epochs, model_name=model_name, device=device, path_save=path_save)

    print(f'Saving {model_name}...')
    save_model(model, f'{path_save}{model_name}/Finale.pl')
    print(f'Done {model_name}!')

print('End of File!')


