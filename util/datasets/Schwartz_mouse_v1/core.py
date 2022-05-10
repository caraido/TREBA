import os
import numpy as np
import torch

from util.datasets import TrajectoryDataset
#from .label_functions import label_functions_list
#from .augmentations import augmentation_list
from util.logging import LogEntry
from .preprocess import *
import pickle

ROOT_DIR='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/data'
TRAIN_FILE='3D_False_train.npz'
TEST_FILE='3D_False_val.npz'
BODYPARTS='3D_False_bodyparts.pk'


class SchwartzMouseV1Dataset(TrajectoryDataset):

    name='Schwartz_mouse_v1'

    # Default config
    _seq_len = 21
    _state_dim = 28
    _action_dim = 28 # TODO: what's the difference?
    _svd_computer=None
    _svd_mean=None

    normalize_data = True
    align_data=True
    compute_svd=False

    test_name=TEST_FILE

    def __init__(self,data_config):
        super().__init__(data_config)

    def _load_data(self):
        # Process configs
        global TRAIN_FILE
        if 'normalize_data' in self.config:
            self.normalize_data = self.config['normalize_data']
        if 'align_data' in self.config:
            self.align_data = self.config['align_data']
        if 'mirror_data' in self.config:
            self.mirror_data=self.config['mirror_data']
        if 'compute_svd' in self.config:
            assert isinstance(self.config['compute_svd'], int)
            self.compute_svd = self.config['compute_svd']
        if 'test_name' in self.config:
            self.test_name = self.config['test_name']

        # this one pick out specific joints to look at (as far as understand)
        # so not used in our case
        self.keypoints = []
        if 'keypoints' in self.config:
            assert isinstance(self.config['keypoints'], list)
            resi_start = [2 * k for k in self.config['keypoints']]
            resi_end = [k + 1 for k in resi_start]
            intr_start = [14 + k for k in resi_start]
            intr_end = [k + 1 for k in intr_start]
            self.keypoints = resi_start + resi_end + intr_start + intr_end
            self.keypoints.sort()

        # also not used here
        if 'labels' in self.config:
            for lf_config in self.config['labels']:
                lf_config['data_align']=self.align_data

        self.log=LogEntry()

        try:
            with open(svd_computer_path, 'rb') as f:
                self._svd_computer = pickle.load(f)
            with open(mean_path, 'rb') as f:
                self._mean = pickle.load(f)
        except:
            pass

        self.train_states, self.train_actions = self._load_and_preprocess(
            train=True)
        self.test_states, self.test_actions = self._load_and_preprocess(
            train=False)

    def _load_and_preprocess(self,train:bool):
        path = os.path.join(ROOT_DIR, TRAIN_FILE if train else self.test_name)
        body_path=os.path.join(ROOT_DIR,BODYPARTS)
        with open(body_path,'rb') as f:
            bodyparts = pickle.load(f)
        file = np.load(path, allow_pickle=True)
        data = file['data']
        if len(data) == 2:
            data = data[0]

        # Subsample timesteps
        data = data[:, ::self.subsample]

        # intepolate nan (move this step to data conversion
        # data= interpol(data)

        # align the data to egocentric
        if self.align_data:
            data,_,_,new_bodyparts = alignment(data,bodyparts=bodyparts)
            # you can only mirror the data for ego centric trajectories
            if self.mirror_data:
                data=mirror(data,bodyparts=new_bodyparts)

        # Select only certain keypoints (NOT USED)
        if len(self.keypoints) > 0:
            data = data[:, :, self.keypoints]

        # Compute SVD on train data and apply to train and test data
        if self.compute_svd:
            seq_len = data.shape[1]

            data = data.reshape((-1, 1, 8, 2))
            # Input [seq_num x seq_len, 1 mouse, 8 bodyparts, 2 xy]
            if train and self._svd_computer is None:
                data_svd, self._svd_computer, self._svd_mean = transform_to_svd_components(
                    data, center_index=3, n_components=self.compute_svd,
                    svd_computer=self._svd_computer, mean=self._svd_mean,
                    stack_agents=True, save_svd=True)
            else:
                data_svd, _, _ = transform_to_svd_components(
                    data, center_index=3, n_components=self.compute_svd,
                    svd_computer=self._svd_computer, mean=self._svd_mean,
                    stack_agents=True)
            # Output [seq_num x seq_len, 2, 4 + n_components]

            data = data_svd.reshape((-1, seq_len, data_svd.shape[-1]))

            states = data
            actions = states[:, 1:] - states[:, :-1]

        else:
            states = data
            actions = states[:, 1:] - states[:, :-1]

        # update dimensions
        self._seq_len=actions.shape[1]
        self._state_dim=states.shape[-1]
        self._action_dim=actions.shape[-1]

        return torch.Tensor(states),torch.Tensor(actions)











