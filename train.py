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

from util.datasets.Schwartz_mouse_v2.preprocess import *

def run_epoch(data_loader, model, device, epoch, train=True, early_break=False, weight_losses=False):
    log = LogEntry()

    # Setting model and dataset into train/eval mode
    if train:
        model = model.train()
        data_loader.dataset.train()
    else:
        model = model.eval()
        data_loader.dataset.eval()

    for batch_idx, (states, actions, labels_dict, ctxt_states, ctxt_actions, ctxt_labels_dict) in tqdm.tqdm(enumerate((data_loader))):
        states = states.to(device).float()
        actions = actions.to(device).float()
        labels_dict = { key: value.to(device) for key, value in labels_dict.items() }
       
        ctxt_states = ctxt_states.to(device).float()
        ctxt_actions = ctxt_actions.to(device).float()
        ctxt_labels_dict = { key: value.to(device).float() for key, value in ctxt_labels_dict.items() }

        if batch_idx > 0 and batch_idx % 10 == 0:
            batch_log = model(
                states,
                actions,
                labels_dict,
                ctxt_states,
                ctxt_actions,
                ctxt_labels_dict,
                restart = True
            )
        else:
            batch_log = model(
                states,
                actions,
                labels_dict,
                ctxt_states,
                ctxt_actions,
                ctxt_labels_dict
            )
 
        # TODO: need to implement this
        #biggest_loss = np.array([loss.detach().cpu().numpy() for loss_name, loss in batch_log.losses.items()]).max()
        #print(batch_log)
        if weight_losses:
            batch_log.losses['nll'] = batch_log.losses['nll'] * 1
            batch_log.losses['triplet'] = batch_log.losses['triplet'] #*3000
            batch_log.losses['quantization'] = batch_log.losses['quantization'] #* 30000000
            batch_log.losses['decoded_LF00_distance_between_ears_Threshold'] = batch_log.losses['decoded_LF00_distance_between_ears_Threshold'] * 2
            batch_log.losses['decoded_LF01_skullbase_to_tailbase_length_Threshold'] = batch_log.losses['decoded_LF01_skullbase_to_tailbase_length_Threshold'] * 2
            batch_log.losses['decoded_LF02_head_body_ratio_Threshold'] = batch_log.losses['decoded_LF02_head_body_ratio_Threshold'] * 2
            batch_log.losses['decoded_LF03_head_body_angle_Threshold'] = batch_log.losses['decoded_LF03_head_body_angle_Threshold'] * 2
            batch_log.losses['decoded_LF04_body_hip_angle_Threshold'] = batch_log.losses[
                                                                             'decoded_LF04_body_hip_angle_Threshold'] * 2
            batch_log.losses['decoded_LF05_speed_Threshold'] = batch_log.losses[
                                                                             'decoded_LF05_speed_Threshold'] * 2
                
        if train:
            model.optimize(batch_log.losses)

        batch_log.itemize() # itemize here since we shouldn't need gradient information anymore
        log.absorb(batch_log)

        if early_break:
            break
            
    log.average(N=len(data_loader.dataset))

    print('TRAIN' if train else 'TEST')
    print(str(log))

    return log.to_dict()


def start_training(save_path, data_config, model_config, train_config, device, test_code=False):
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
    model_config['num_bins']=data_config['num_bins']

    #if model_class.requires_augmentations:
    model_config['augmentations'] = dataset.active_augmentations
        
    # Initialize model
    model = model_class(model_config).to(device)
    summary['model'] = model_config
    summary['model']['num_parameters'] = model.num_parameters

    # Initialize dataloaders
    kwargs = {'num_workers': 10, 'pin_memory': False, 'worker_init_fn': np.random.seed(seed)} if device != 'cpu' else {}
    data_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, **kwargs)

    # Initialize with pretrained model (if specified)
    if 'pretrained_model' in train_config:
        print('LOADING pretrained model: {}'.format(train_config['pretrained_model']))
        # model_path = os.path.join(os.path.dirname(save_path), train_config['pretrained_model'])
        model_path = os.path.join(os.path.dirname(os.path.dirname(save_path)), train_config['pretrained_model'])
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth')) # copy over best model

    # Start training
    if isinstance(train_config['num_epochs'], int):
        train_config['num_epochs'] = [train_config['num_epochs']]

    start_time = time.time()
    epochs_done = 0




    # TEMPORARY
    PICK_BEST_TRIPLET = True
    WEIGHT_LOSSES = True




    for num_epochs in train_config['num_epochs']:

        model.prepare_stage(train_config)
        stage_start_time = time.time()
        print('##### STAGE {} #####'.format(model.stage))

        best_test_log = {}
        best_test_log_times = []
        
        for epoch in range(num_epochs):
            epochs_done += 1
            print('--- EPOCH [{}/{}] ---'.format(epochs_done, sum(train_config['num_epochs'])))

            epoch_start_time = time.time()
            train_log = run_epoch(data_loader, model, device, epoch, weight_losses=WEIGHT_LOSSES, train=True, early_break=test_code)
            test_log = run_epoch(data_loader, model, device, epoch, weight_losses=WEIGHT_LOSSES, train=False, early_break=test_code)
            epoch_time = time.time() - epoch_start_time
            print('{:.3f} seconds'.format(epoch_time))
        
            logger.append({
                'epoch' : epochs_done,
                'stage' : model.stage,
                'train' : train_log,
                'test' : test_log,
                'time' : epoch_time
                })
        
            # Save model checkpoints
            if epochs_done % train_config['checkpoint_freq'] == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'checkpoints', 'checkpoint_{}.pth'.format(epochs_done)))
                print('Checkpoint saved')

            # Save model with best test loss during stage

            if PICK_BEST_TRIPLET:
                if epoch == 0 or (test_log['losses']['triplet']) < (best_test_log['losses']['triplet']):
                    best_test_log = test_log
                    best_test_log_times.append(epochs_done)
                    torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
                    print('Best model saved')
            else:
                if epoch == 0 or sum(test_log['losses'].values()) < sum(best_test_log['losses'].values()):
                    best_test_log = test_log
                    best_test_log_times.append(epochs_done)
                    torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
                    print('Best model saved')


        # Save training statistics by stage
        summary['training'].append({
            'stage' : model.stage,
            'num_epochs' : num_epochs,
            'stage_time' : round(time.time()-stage_start_time, 3),
            'best_test_log_times' : best_test_log_times,
            'best_test_log' : best_test_log
            })

        # Load best model for next stage
        if model.stage < len(train_config['num_epochs']):
            best_state = torch.load(os.path.join(save_path, 'best.pth'))
            model.load_state_dict(best_state)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_stage_{}.pth'.format(model.stage)))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path,'final.pth'))
    print('Final model saved')

    # Save total time
    summary['total_time'] = round(time.time()-start_time, 3)

    model_config.pop('label_functions')
    model_config.pop('augmentations')

    return summary, logger, data_config, model_config, train_config
