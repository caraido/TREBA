import torch
import os
import numpy as np
import json
import scipy.io as sio
import pickle as pk

from torch.utils.data import DataLoader
from util.datasets import load_dataset
from lib.models import get_model_class
from generate_embeddings_reconstruction import get_model

def extract_features(config):
    #print('#################### Feature Extraction {} ####################'.format(trial_id))
    data_config=config['data_config']
    # No need to load training data for feature extraction.
    data_config['label_train_set']=False

    dataset=load_dataset(data_config)
    all_states=dataset.test_states[:,0,:]
    label_feature_dict={}
    for i , lf in enumerate(dataset.active_label_functions):
        label=lf.label_func(states=all_states,
                            actions=None,
                            true_label=None,
                            full=True)
        label_feature_dict[lf.name]=label.detach().numpy()

    return label_feature_dict



# this part extract features
if __name__=='__main__':
    config_name='run_7'
    config_path=f'/home/roton2/PycharmProjects/TREBA/configs/Schwartz_mouse/{config_name}.json'
    #test_name='cagmates/3D_False_train_231'

    with open(config_path,'r') as f:
        config=json.load(f)


    #id=231
    #foldername = f'cagemates/3D_False_train_{id}'
    path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed'
    type='all'
    path=os.path.join(path,type)
    # foldername=f'all/3D_False_all'
    # foldername=f'cagemates/{name[:-3]}'
    for item in os.listdir(path):
        foldername=os.path.join(type,item)
        config['data_config']['filename'] = f'{foldername}.pk'
        config['data_config']['subsample'] = 1 # comment this if the dataset is all/3D_False_train
        config['data_config']['label_train_set'] = False
        config['data_config']['val_prop']=1
        config['data_config']['root_data_directory']="/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/data"
        config['data_config'][
            'train_labels_path'] = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/labels/{foldername}/train_labels.json'
        config['data_config'][
            'test_labels_path'] = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/labels/{foldername}/test_labels.json'
        #config['data_config']['traj_len'] = 20
        config['data_config'][
            'ctxt_test_labels_path'] = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/labels/{foldername}/ctxt_test_labels.json'
        config['data_config'][
            'ctxt_train_labels_path'] = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/labels/{foldername}/ctxt_train_labels.json'

        labels=extract_features(config)
        save_root_dir='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2'

        #save_dict={}
        file_path = os.path.join(save_root_dir, 'features', foldername,config_name)
        #for name, label in labels.items():

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        #    save_dict[name]=label
            #sio.savemat(os.path.join(file_path,name+'.mat'),{name:label})

        with open(os.path.join(file_path,'features.pk'),'wb') as f:
            pk.dump(labels,f)
        print(f'feature saved for {file_path}!')




