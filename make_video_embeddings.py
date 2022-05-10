import os
import json
import torch
import numpy as np
from util.datasets import load_dataset
from lib.models import get_model_class
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cpu')

def load_model(dataset, model_config):
    # Load the model
    # Add state and action dims to model config 
        
    ## WITH REMOVING CENTROIDS AND ABSOLUTE ANGLES

    model_config['state_dim'] = dataset.state_dim
    model_config['action_dim'] = dataset.action_dim 

    # Get model class
    model_class = get_model_class(model_config['name'].lower())

    # Check if model needs labels as input
    # if model_class.requires_labels:
    model_config['label_dim'] = dataset.label_dim
    model_config['label_functions'] = dataset.active_label_functions


    # if model_class.requires_augmentations:
    model_config['augmentations'] = dataset.active_augmentations

    # Initialize model
    model = model_class(model_config).to(device)

    return model

if __name__ == "__main__":

#    root_dir = '/home/andrewulmer/data/parker_lab/Clozapine/Amph/'
#    vid_list = [
#        "20191205_m972_clo01_amph",
#        "20191205_m085_clo01_amph",
#        "20200722_m404_clo01_amph",
#        "20200518_m380_clo01_amph",
#        "20200224_m106_clo01_amph",
#        "20210104_f857_clo01_amph",
#        "20200722_f414_clo01_amph",
#        "20210104_m794_clo01_amph",
#        "20200110_m040_clo01_amph",
#        "20210104_m797_clo01_amph",
#        "20200916_m483_clo01_amph",
#        "20200518_m377_clo01_amph",
#        "20200407_m298_clo01_amph",
#        "20210107_m795_clo01_amph",
#        "20200916_f487_clo01_amph",
#        "20201026_f694_clo01_amph",
#        "20191205_m971_clo01_amph",
#        "20200916_f480_clo01_amph",
#        "20200916_m485_clo01_amph",
#        "20210104_f859_clo01_amph"
#    ]

    root_dir = '/home/andrewulmer/data/parker_lab/Clozapine/Control/'
    vid_list = [
        "20200407_m298_clo01",
        "20210104_m794_clo01",
        "20200110_m040_clo01",
        "20210104_m797_clo01",
        "20210107_m795_clo01",
        "20191205_m972_clo01",
        "20200916_m485_clo01",
        "20200518_m377_clo01",
        "20200916_m483_clo01",
        "20200916_f487_clo01",
        "20200916_f480_clo01",
        "20200722_m404_clo01",
        "20200518_m380_clo01",
        "20191205_m085_clo01",
        "20210104_f857_clo01",
        "20201026_f694_clo01",
        "20200224_m106_clo01",
        "20210104_f859_clo01",
        "20200722_f414_clo01",
        "20191205_m971_clo01"
    ]

    
    # Load the configuration file
    full_config_path = './configs/vq-triplet-weighted-evenly-2/run_2.json'
    with open(full_config_path, 'r') as myFile:
        full_config = json.load(myFile)

    for vid in vid_list:
        print(f'Making embeddings for {vid}')

        # Update configuration to match video that we're on
        vid_name = vid.strip('"').strip('/').strip('./')
        full_path = os.path.join(root_dir)
        full_config['data_config']['root_data_directory'] = full_path
        full_config['data_config']['new_video'] = True
        full_config['data_config']['subset'] = [vid]

        # Create a dataset object using the new configuration
        full_config['data_config']['test_save_path'] = os.path.join(root_dir, vid, 'test_labels.pkl')
        full_config['data_config']['ctxt_test_save_path'] = os.path.join(root_dir, vid, 'test_ctxt_labels.pkl')
        full_config['data_config']['svd_computer_path'] = '/home/andrewulmer/code/TRIPLET-TREBA-2/util/datasets/mouse_v2/svd/svd.pickle'
        full_config['data_config']['mean_computer_path'] = '/home/andrewulmer/code/TRIPLET-TREBA-2/util/datasets/mouse_v2/svd/mean.pickle'
        full_config['data_config']['val_prop'] = 1.0

        dataset = load_dataset(full_config['data_config'])

        # Load the model
        model_config = full_config['model_config']
        model = load_model(dataset, model_config)
        model_name = full_config_path.split('/')[2]
        state_dict = torch.load(os.path.join(f"./saved/{model_name}/run_1/best.pth"), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        # Generate embeddings for the video
        dataset.eval()
        data_loader = DataLoader(dataset, batch_size = full_config['train_config']['batch_size'], shuffle=False)
        embeddings, p_to_as, n_to_as = [], [], []
        for batch_idx, (states, actions, labels_dict, ctxt_states, ctxt_actions, ctxt_labels_dict) in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                states = states.to(device).float()
                actions = actions.to(device).float()
                labels_dict = {key: value.to(device) for key, value in labels_dict.items()}

                ctxt_states = ctxt_states.to(device).float()
                ctxt_actions = ctxt_actions.to(device).float()
                ctxt_labels_dict = {key: value.to(device).float() for key, value in ctxt_labels_dict.items()}

                _, embedding, = model(
                    states,
                    actions,
                    labels_dict,
                    ctxt_states,
                    ctxt_actions,
                    ctxt_labels_dict,
                    embed=True
                )

                embeddings.append(embedding)

        embeddings = torch.cat(embeddings).numpy().reshape(-1)
        save_path = os.path.join(full_path, vid_name, f'{model_name}_clusts.npz')
        np.savez(save_path, data=embeddings)
