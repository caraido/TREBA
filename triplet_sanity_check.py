import os
import random
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util.datasets import load_dataset
from lib.models import get_model_class
from util.logging import LogEntry

from util.datasets.mouse_v2.preprocess import *

def run_epoch(data_loader, model, device, train=True, early_break=False, weight_losses=False):
    log = LogEntry()

    # Setting model and dataset into train/eval mode
    model = model.train()
    data_loader.dataset.train()
    
    p_to_as, n_to_as = [], []

    with torch.no_grad():
        for batch_idx, (states, actions, labels_dict, ctxt_states, ctxt_actions, ctxt_labels_dict) in enumerate(tqdm(data_loader)):
            states = states.to(device).float()
            actions = actions.to(device).float()
            labels_dict = { key: value.to(device) for key, value in labels_dict.items() }
           
            ctxt_states = ctxt_states.to(device).float()
            ctxt_actions = ctxt_actions.to(device).float()
            ctxt_labels_dict = { key: value.to(device).float() for key, value in ctxt_labels_dict.items() }
     
            _, _, p_to_a, n_to_a = model(states, actions, labels_dict, ctxt_states, ctxt_actions, ctxt_labels_dict, embed=True)
            p_to_as.append(p_to_a.detach().cpu().numpy())
            n_to_as.append(n_to_a.detach().cpu().numpy())
    
    p_to_as = np.concatenate(p_to_as)
    n_to_as = np.concatenate(n_to_as)

    return p_to_as, n_to_as



def sanity_check(save_path, data_config, model_config, train_config, device, test_code=False):
    summary = { 'training' : [] }
    logger = []

    # Sample and fix a random seed if not set in train_config
    if 'seed' not in train_config:
        train_config['seed'] = random.randint(0, 9999)
    seed = train_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize dataset
    dataset = load_dataset(data_config)
    dataset.eval()
    summary['dataset'] = dataset.summary

    # Add state and action dims to model config
    model_config['state_dim'] = dataset.state_dim     # REMOVING ABSOLUTE CENTROID POSITION + ANGLE
    model_config['action_dim'] = dataset.action_dim   # REMOVING ABSOLUTE CENTROID POSITION + ANGLE

    # Get model class
    model_class = get_model_class(model_config['name'].lower())

    # Check if model needs labels as input
    #if model_class.requires_labels:
    model_config['label_dim'] = dataset.label_dim
    model_config['label_functions'] = dataset.active_label_functions

    #if model_class.requires_augmentations:
    model_config['augmentations'] = dataset.active_augmentations
        
    # Initialize model
    model = model_class(model_config).to(device)
    summary['model'] = model_config
    summary['model']['num_parameters'] = model.num_parameters

    # Initialize dataloaders
    kwargs = {'num_workers': 8, 'pin_memory': False, 'worker_init_fn': np.random.seed(seed)} if device is not 'cpu' else {}
    data_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, **kwargs)


    model_path = './saved/triplet-treba-margin-0.1-registered-2/run_1/best.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), os.path.join(save_path, 'best.pth')) # copy over best model


    # TEMPORARY
    PICK_BEST_TRIPLET = False
    WEIGHT_LOSSES = False


    model.prepare_stage(train_config)

    _, p_to_as, n_to_as = run_epoch(data_loader, model, device, weight_losses=WEIGHT_LOSSES, train=True)
    

    return summary, logger, data_config, model_config, train_config
