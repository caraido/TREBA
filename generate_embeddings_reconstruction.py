import json
import os
import numpy as np
import shutil

from util.datasets import load_dataset
from util.datasets.Schwartz_mouse_v1.core import ROOT_DIR
import torch
from lib.models import get_model_class
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pk
from util.datasets.Schwartz_mouse_v1.preprocess import svd_computer_path,mean_path, transform_svd_to_keypoints


def reconstruction(data_compound,model):

    new_states=[]
    all_embeddings=[]
    for idx,data in tqdm(enumerate(data_compound)):
        states,actions=data
        states=states.unsqueeze(0)
        actions=actions.unsqueeze(0)
        states=states.to(device)
        actions=actions.to(device)

        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        # encode
        posterior=model.encode(states[:-1],actions=actions,labels=None)
        embedding=posterior.mean

        model.reset_policy(labels=None, z=posterior.sample())

        # decode
        actions_hat=[]
        states_hat=[states[0]]
        new_state=0

        # get states estimation from the posterior distribution
        for t in range(actions.size(0)):
            # decode action
            new_action=model.act(states[t])
            # reconstruct state from the action
            if t==0:
                new_state= states[0]+new_action
            else:
                new_state=new_state+new_action
            actions_hat.append(new_action)
            states_hat.append(new_state)

        states_hat=torch.stack(states_hat)
        new_states.append(torch.squeeze(states_hat).cpu().detach().numpy())
        all_embeddings.append(torch.squeeze(embedding).cpu().detach().numpy())

    new_states=np.array(new_states)
    all_embeddings=np.array(all_embeddings)

    return all_embeddings,new_states

def get_data(config):
    # load data to the dataloader
    dataset=load_dataset(config)
    data_loader=DataLoader(dataset,batch_size=1,shuffle=False)
    return dataset, data_loader

def get_model(config,dataset):
    model_config = config
    model_config['state_dim'] = dataset.state_dim
    model_config['action_dim'] = dataset.action_dim
    model_class = get_model_class(model_config['name'].lower())
    model = model_class(model_config).to(device)
    return model

def prepare_model(model, state_dict):
    model.load_state_dict(state_dict)
    # double check if using model.eval is correct
    model=model.eval()
    return model

def generate_embeddings_reconstruction(config,train,test):
    dataset, data_loader = get_data(config['data_config'])
    # get model config and initialize the model
    model = get_model(config['model_config'],dataset)

    # prepare model
    state_dict = torch.load(best_result_path)
    model = prepare_model(model, state_dict)
    data_loader.dataset.eval()
    # assign the best model to it

    # calculating embeddings and reconstructed trajectories for train data
    if train:
        train_data = zip(dataset.train_states, dataset.train_actions)
        train_embeddings, train_reconstructed = reconstruction(train_data, model)
        train_original = dataset.train_states.cpu().detach().numpy()

    if test:
        test_data = zip(dataset.test_states, dataset.test_actions)
        test_embeddings, test_reconstructed = reconstruction(test_data, model)
        test_original = dataset.test_states.cpu().detach().numpy()

    # if we used svd, here we need to inverse transform the data into trajectories
    if 'compute_svd' in config['data_config']:
        global svd_computer_path
        global mean_path
        svd_computer_path = os.path.join(root_dir, svd_computer_path)
        mean_path = os.path.join(root_dir, mean_path)
        with open(svd_computer_path, 'rb') as f:
            svd_computer = pk.load(f)
        with open(mean_path, 'rb') as f:
            bp_mean = pk.load(f)

        # for train data
        if train:
            train_shape = train_reconstructed.shape
            train_reconstructed = train_reconstructed.reshape(-1, train_shape[-1])
            train_reconstructed = transform_svd_to_keypoints(train_reconstructed, svd_computer, bp_mean)
            train_reconstructed = train_reconstructed.reshape(train_shape[0], train_shape[1], -1)

            train_original = train_original.reshape(-1, train_shape[-1])
            train_original = transform_svd_to_keypoints(train_original, svd_computer, bp_mean)
            train_original = train_original.reshape(train_shape[0], train_shape[1], -1)

        # for test data
        if test:
            test_shape = test_reconstructed.shape
            test_reconstructed = test_reconstructed.reshape(-1, test_shape[-1])
            test_reconstructed = transform_svd_to_keypoints(test_reconstructed, svd_computer, bp_mean)
            test_reconstructed = test_reconstructed.reshape(test_shape[0], test_shape[1], -1)

            test_original = test_original.reshape(-1, test_shape[-1])
            test_original = transform_svd_to_keypoints(test_original, svd_computer, bp_mean)
            test_original = test_original.reshape(test_shape[0], test_shape[1], -1)

    # here we have (1)new states, (2)old states, (3)best_parameters, (4)embeddings
    # now we need to save them. (1) should have the same format with (2)
    # best_parameters information should be stored at....
    # embeddings should be np.array
    # meta data: model, config folder, data path information

    # load model run config
    with open(run_config_path, 'r') as f:
        run_config = json.load(f)

    # add information of the test dataset
    run_config['test_dataset'] = config['data_config']

    project_name = config['data_config']['name']
    save_root_path = os.path.join(dataset_path, project_name, 'reconstructed')

    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    test_name = config['data_config']['test_name']
    save_path = os.path.join(save_root_path, config['data_config']['test_name'][:-4])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # save the model,dataset,test dataset, training information
    information_path = os.path.join(save_path, 'information.json')
    with open(information_path, 'w') as f:
        json.dump(run_config, f, indent=4)
    print('information saved')

    if test:
        # save original test data
        test_original_path = os.path.join(save_path, 'original_all.npy')
        np.save(test_original_path, test_original)
        print('original test data saved')

        # save reconstructed test data
        test_reconstructed_path = os.path.join(save_path, 'reconstructed_all.npy')
        np.save(test_reconstructed_path, test_reconstructed)
        print('reconstructed test data saved')

        # save the test embeddings
        embedding_path = os.path.join(save_path, 'embeddings_all.npy')
        np.save(embedding_path, test_embeddings)
        print('test embeddings saved')

    if train:
        # save original train data
        train_original_path = os.path.join(save_path, 'original_train.npy')
        np.save(train_original_path, train_original)
        print('original train data saved')

        # save reconstructed train data
        train_reconstructed_path = os.path.join(save_path, 'reconstructed_train.npy')
        np.save(train_reconstructed_path, train_reconstructed)
        print('reconstructed train data saved')

        # save the train embeddings
        embedding_path = os.path.join(save_path, 'embeddings_train.npy')
        np.save(embedding_path, train_embeddings)
        print('train embeddings saved')

if __name__=='__main__':
    device='cuda:0'
    config_path='/home/roton2/PycharmProjects/TREBA/configs/Schwartz_mouse/apply.json'

    dataset_path= "/home/roton2/PycharmProjects/TREBA/util/datasets"
    best_result_path='/home/roton2/PycharmProjects/TREBA/saved/Schwartz_mouse/run/best.pth'
    run_config_path= '/home/roton2/PycharmProjects/TREBA/saved/Schwartz_mouse/run/summary.json'

    root_dir='/home/roton2/PycharmProjects/TREBA'
    train=False
    test=True

    with open(config_path, 'r') as f:
        config = json.load(f)

    session_idx=np.arange(0,24)
    for idx in session_idx:
        config['data_config']['test_name']=f'3D_False_idx_{idx}_test.npz'
        generate_embeddings_reconstruction(config,train,test)
        print(f'saved session id: {idx}')















