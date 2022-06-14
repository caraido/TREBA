import json
import random
import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np


TRAIN = 1
EVAL = 2


class TrajectoryDataset(Dataset):

    # Default parameters
    mode = TRAIN
    subsample = 1

    _state_dim = 0
    _action_dim = 0
    _seq_len = 0

    # Ground-truth labels
    hasTrueLabels = False
    train_labels = None
    test_labels = None

    label_train_set = True
    bodyparts=None

    def __init__(self, data_config):
        assert hasattr(self, 'name')

        # Check for labels and labeling functions
        if 'labels' in data_config:
            assert isinstance(data_config['labels'], list)

            if len(data_config['labels']) > 0:
                assert hasattr(self, 'all_label_functions') and len(self.all_label_functions) > 0
        else:
            data_config['labels'] = []

        if 'seq_len' in data_config:
            assert isinstance(data_config['seq_len'],int)
            self._seq_len=data_config['seq_len']

        # Check for augmentations
        if 'augmentations' in data_config:
            assert isinstance(data_config['augmentations'], list)

            if len(data_config['augmentations']) > 0:
                assert hasattr(self, 'all_augmentations') and len(self.all_augmentations) > 0
        else:
            data_config['augmentations'] = []


        # Check if trajectories will be subsampled
        if 'subsample' in data_config:
            assert isinstance(data_config['subsample'], int) and data_config['subsample'] > 0
            self.subsample = data_config['subsample']

        self.config = data_config
        if 'ctxt_test_save_path' not in self.config:
            self.config['ctxt_test_save_path'] = f'./util/datasets/{self.name}/labels/ctxt_test_labels.json'
        if 'test_save_path' not in self.config:
            self.config['test_save_path'] = f'./util/datasets/{self.name}/labels/test_labels.json'

        self.summary = {'name' : self.name}

        if 'label_train_set' in data_config:
            self.label_train_set = data_config['label_train_set']

        # Load data (and true labels, if any)
        self._load_data()

        # Assertions for train data
        assert hasattr(self, 'train_states') and isinstance(self.train_states, torch.Tensor)
        assert hasattr(self, 'train_actions') and isinstance(self.train_actions, torch.Tensor)
        assert hasattr(self, 'train_ctxt_states') and isinstance(self.train_ctxt_states, torch.Tensor)
        assert hasattr(self, 'train_ctxt_actions') and isinstance(self.train_ctxt_actions, torch.Tensor)
        assert self.train_states.size(0) == self.train_actions.size(0)
        assert self.train_ctxt_states.size(0) == self.train_ctxt_actions.size(0)
        assert self.train_states.size(1) - 1 == self.train_actions.size(1) == self.seq_len
        assert self.train_ctxt_states.size(2) - 1 == self.train_ctxt_actions.size(2) == self.seq_len

        # Assertions for test data
        assert hasattr(self, 'test_states') and isinstance(self.test_states, torch.Tensor)
        assert hasattr(self, 'test_actions') and isinstance(self.test_actions, torch.Tensor)
        assert hasattr(self, 'test_ctxt_states') and isinstance(self.test_ctxt_states, torch.Tensor)
        assert hasattr(self, 'test_ctxt_actions') and isinstance(self.test_ctxt_actions, torch.Tensor)
        assert self.test_states.size(0) == self.test_actions.size(0)
        assert self.test_ctxt_states.size(0) == self.test_ctxt_actions.size(0)
        assert self.test_states.size(1) - 1 == self.test_actions.size(1) == self.seq_len
        assert self.test_ctxt_states.size(2) - 1 == self.test_ctxt_actions.size(2) == self.seq_len

        # Assertions for ground-truth labels
        if self.hasTrueLabels:
            assert isinstance(self.train_labels, torch.Tensor)
            assert self.train_labels.size(0) == self.train_data.size(0)
            assert isinstance(self.test_labels, torch.Tensor)
            assert self.test_labels.size(0) == self.test_data.size(0)
        
        # Apply augmentations
        self._init_augmentations()        
        # Apply labeling functions: already finished in load data

        if self.config['new_threshold']:
            new_threshold_path=f'./util/datasets/{self.name}/labels/label_threshold.json'
            try:
                new_threshold_path=self.config['new_threshold_path']
            except:
                pass
            if not os.path.exists(os.path.split(new_threshold_path)[0]):
                os.makedirs(os.path.split(new_threshold_path)[0])
            with open(new_threshold_path,'w') as f:
                thresholds_dict=[{'name':a['name'],'thresholds':a['thresholds']} for a in self.config['labels']]
                json.dump({'labels':thresholds_dict},f,indent=4)


        train_save_path = f'./util/datasets/{self.name}/labels/train_labels.json'
        test_save_path = f'./util/datasets/{self.name}/labels/test_labels.json'
        try:
            train_save_path = self.config['train_labels_path']
        except:
            pass
        try:
            test_save_path = self.config['test_labels_path']
        except:
            pass

        if self.label_train_set:
            # TODO: need to move this out
            print("labeling the train set.")
            if os.path.exists(train_save_path):
                with open(train_save_path, 'rb') as myfile:
                    self.train_lf_labels = pickle.load(myfile)
            else:
                self.train_lf_labels = self.label_data(self.train_states,
                                                       self.train_actions,
                                                       self.train_labels)
                if not os.path.exists(os.path.split(train_save_path)[0]):
                    os.makedirs(os.path.split(train_save_path)[0])
                with open(train_save_path, 'wb') as myfile:
                    pickle.dump(self.train_lf_labels, myfile)
        else:
            self.train_lf_labels = []

        if os.path.exists(test_save_path):
            print("Loading test set")
            with open(test_save_path, 'rb') as myfile:
                self.test_lf_labels = pickle.load(myfile)
        else:
            print("Labeling test set")
            self.test_lf_labels = self.label_data(self.test_states,
                                                  self.test_actions,
                                                  self.test_labels)
            if not os.path.exists(os.path.split(test_save_path)[0]):
                os.makedirs(os.path.split(test_save_path)[0])
            with open(test_save_path, 'wb') as myfile:
                pickle.dump(self.test_lf_labels, myfile)

        # MAY NEED CONTEXT LABELS
        ctxt_train_save_path = f'./util/datasets/{self.name}/labels/ctxt_train_labels.json'
        ctxt_test_save_path = f'./util/datasets/{self.name}/labels/ctxt_test_labels.json'
        try:
            ctxt_train_save_path=self.config['ctxt_train_labels_path']
        except:
            pass

        try:
            ctxt_test_save_path=self.config['ctxt_test_labels_path']
        except:
            pass

        if self.label_train_set:
            if os.path.exists(ctxt_train_save_path):
                with open(ctxt_train_save_path, 'rb') as myfile:
                    self.train_ctxt_lf_labels = pickle.load(myfile)
            else:
                self.train_ctxt_lf_labels = self.label_context_data(self.train_ctxt_states,
                                                                    self.train_ctxt_actions,
                                                                    None)
                if not os.path.exists(os.path.split(ctxt_train_save_path)[0]):
                    os.makedirs(os.path.split(ctxt_train_save_path)[0])
                with open(ctxt_train_save_path, 'wb') as myfile:
                    pickle.dump(self.train_ctxt_lf_labels, myfile)
        else:
            self.train_ctxt_lf_labels = []

        if os.path.exists(ctxt_test_save_path):
            print("Loading test set context")
            with open(ctxt_test_save_path, 'rb') as myfile:
                self.test_ctxt_lf_labels = pickle.load(myfile)
        else:
            print('Labeling test set context')
            self.test_ctxt_lf_labels = self.label_context_data(self.test_ctxt_states,
                                                               self.test_ctxt_actions,
                                                               None)
            if not os.path.exists(os.path.split(ctxt_test_save_path)[0]):
                os.makedirs(os.path.split(ctxt_test_save_path)[0])
            with open(ctxt_test_save_path, 'wb') as myfile:
                pickle.dump(self.test_ctxt_lf_labels, myfile)

        # Compute statistics for label distributions
        for lf in self.active_label_functions:
            if lf.categorical:
                if self.label_train_set:
                    train_dist = torch.mean(self.train_lf_labels[lf.name].float(), dim=0)
                    self.summary['label_functions'][lf.name]['train_dist'] = train_dist.squeeze().tolist()
                # print(torch.mean(self.train_lf_labels[lf.name][:, :, 0].float()))
                test_dist = torch.mean(self.test_lf_labels[lf.name].float(), dim=0)
                self.summary['label_functions'][lf.name]['test_dist'] = test_dist.squeeze().tolist()
                # print(torch.mean(self.test_lf_labels[lf.name][:, :, 0].float()))
            else:
                train_labels = self.train_lf_labels[lf.name]
                self.summary['label_functions'][lf.name]['train_dist'] = {
                    'min' : torch.min(train_labels).item(),
                    'max' : torch.max(train_labels).item(),
                    'mean' : torch.mean(train_labels.float()).item()
                }

                test_labels = self.test_lf_labels[lf.name]
                self.summary['label_functions'][lf.name]['test_dist'] = {
                    'min' : torch.min(test_labels).item(),
                    'max' : torch.max(test_labels).item(),
                    'mean' : torch.mean(test_labels.float()).item()
                }

        self.states = { TRAIN : self.train_states, EVAL : self.test_states }
        self.actions = { TRAIN : self.train_actions, EVAL : self.test_actions }
        self.lf_labels = { TRAIN : self.train_lf_labels, EVAL : self.test_lf_labels }

        self.ctxt_states = {TRAIN: self.train_ctxt_states, EVAL: self.test_ctxt_states}
        self.ctxt_actions = {TRAIN: self.train_ctxt_actions, EVAL: self.test_ctxt_actions}
        self.ctxt_lf_labels = {TRAIN: self.train_ctxt_lf_labels, EVAL: self.test_ctxt_lf_labels}

    def __len__(self):
        return self.states[self.mode].size(0)

    def __getitem__(self, index):
        states = self.states[self.mode][index,:,:self.state_dim]
        actions = self.actions[self.mode][index,:,:self.action_dim]
        labels_dict = { key: val[index] for key, val in self.lf_labels[self.mode].items() }

        ctxt_states = self.ctxt_states[self.mode][index]
        ctxt_actions = self.ctxt_actions[self.mode][index]
        ctxt_labels_dict = {key: val[index] for key, val in self.ctxt_lf_labels[self.mode].items()}

        return states, actions, labels_dict, ctxt_states, ctxt_actions, ctxt_labels_dict

    @property
    def seq_len(self):
        assert self._seq_len > 0
        return self._seq_len

    @property
    def state_dim(self):
        assert self._state_dim > 0
        return self._state_dim

    @property
    def action_dim(self):
        assert self._action_dim > 0
        return self._action_dim
        
    def _load_data(self):
        raise NotImplementedError

    def train(self):
        self.mode = TRAIN

    def eval(self):
        self.mode = EVAL

    def _get_label_function(self, lf_config):
        lf_name = lf_config['name'].lower()

        for lf in self.all_label_functions:
            if lf.name == lf_name:
                return lf(lf_config,self.bodyparts)

        raise NotImplementedError

    def _get_augmentations(self, aug_config):
        aug_name = aug_config['name'].lower()

        for aug in self.all_augmentations:
            if aug.name == aug_name:
                return aug(aug_config)

        raise NotImplementedError


    def _init_augmentations(self):
        self.active_augmentations = []
        self.summary['augmentations'] = {}
        Augmentations.Counter = 0 # reset counter for augmentations

        for aug_config in self.config['augmentations']:
            aug_config['svd_computer'] = self._svd_computer
            aug_config['mean'] = self._mean
            aug = self._get_augmentations(aug_config)
            self.active_augmentations.append(aug)
            self.summary['augmentations'][aug.name] = aug.summary


    def _init_label_functions(self):
        self.label_dim = 0
        self.active_label_functions = []
        self.summary['label_functions'] = {}
        LabelFunction.Counter = 0 # reset counter for labeling functions

        for lf_config in self.config['labels']:
            lf_config['svd_computer'] = self._svd_computer
            lf_config['mean'] = self._svd_mean
            lf = self._get_label_function(lf_config)
            self.label_dim += lf.output_dim
            self.active_label_functions.append(lf)
            self.summary['label_functions'][lf.name] = lf.summary

    def label_data(self, states, actions, true_labels=None):
        """Labels a batch of data."""
        labels = {}

        for lf in self.active_label_functions:
            labels[lf.name] = lf.label(states, actions, true_labels, batch=True)
            assert labels[lf.name].size(0) == states.size(0) == actions.size(0)

        return labels

    def label_context_data(self, context_states, context_actions, true_labels=None):
        """Labels a batch of data."""
        labels = {}

        context_states, context_actions = context_states.transpose(1, 0), context_actions.transpose(1, 0)
        for lf in self.active_label_functions:
            curr_lbl = []
            for states, actions in zip(context_states, context_actions):
                print('something')
                curr_lbl.append(lf.label(states, actions, true_labels, batch=True))

            labels[lf.name] = torch.stack(curr_lbl, axis=1)

        return labels

    def generate_random_labels(self, num_labels):
        random_labels = {}

        for lf in self.active_label_functions:
            random_labels[lf.name] = lf.generate_random_labels(num_labels)

        return random_labels
    

class LabelFunction(object):

    Counter = 0

    def __init__(self, lf_config, output_dim):
        assert hasattr(self, 'name')
        self.name = 'LF{:02d}_{}'.format(LabelFunction.Counter, self.name)
        LabelFunction.Counter += 1

        assert output_dim > 0

        self.config = lf_config
        self.output_dim = output_dim

        if not hasattr(self, 'categorical'):
            self.categorical = False

        if 'thresholds' in lf_config:
            assert self.output_dim == 1 # can only apply thresholds on single values
            assert isinstance(lf_config['thresholds'], list)
            assert len(lf_config['thresholds']) > 0
            self.thresholds = torch.tensor(sorted(lf_config['thresholds'])).float()
            self.output_dim = self.thresholds.size(0) + 1
            self.categorical = True
            self.name += '_Threshold'

        self.summary = {
            'output_dim' : self.output_dim,
            'categorical' : self.categorical
            }

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def create_negative_bins(self, labels, pmf_bins=5000):
        # Density doesn't make a pdf sum to 1 unless all the bin widths = 1
        values, edges = np.histogram(labels, bins=pmf_bins)
        pdf = values / values.sum()
        cdf = np.array([pdf[:i].sum() for i, _ in enumerate(pdf)])

        # Using 5 bins now
        neg_bins = np.vstack([np.array([0.2 * i for i in range(1, 6)])] * pmf_bins)
        diff = np.abs(neg_bins - np.expand_dims(cdf, axis=1))
        closest = np.argmin(diff, axis=0)
        thresholds = edges[closest]

    def label(self, states, actions, true_labels=None, batch=False, apply_threshold=True):
        # Some preprocessing
        if not batch:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            true_labels = [None] if true_labels is None else true_labels.unsqueeze(0)
        elif true_labels is None:
            true_labels = [None]*states.size(0)

        assert len(true_labels) == states.size(0) == actions.size(0)

        # Apply labeling function
        labels = []
        for i in range(states.size(0)):
            label = self.label_func(states[i], actions[i], true_labels[i])
            assert isinstance(label, torch.Tensor)
            labels.append(label.unsqueeze(0))
        labels = torch.cat(labels, dim=0)

        if hasattr(self, 'thresholds') and apply_threshold:
            # Threshold values
            assert len(labels.size()) == 1
            labels = torch.sum(self.thresholds < labels.unsqueeze(1), dim=1)
            labels = self._one_hot_encode(labels.long())
        elif self.categorical and len(labels.size()) == 1:
            # Convert to one-hot-encodings
            assert len(labels.size()) == 1
            labels = self._one_hot_encode(labels.long())
        elif self.categorical:
            # Convert to one-hot-encodings
            labels = torch.nn.functional.one_hot(labels.long(), num_classes = 2).float()

        if self.output_dim == 1:
            labels = labels.unsqueeze(-1)

        return labels

    def label_func(self, states, actions, true_label=None,full=False):
        raise NotImplementedError


    def _one_hot_encode(self, labels):
        dims = [labels.size(i) for i in range(len(labels.size()))]
        dims.append(self.output_dim)
        label_ohe = torch.zeros(dims)
        label_ohe.scatter_(-1, labels.unsqueeze(-1), 1)
        return label_ohe

    def generate_random_labels(self, num_labels):
        import pdb; pdb.set_trace()
        random_labels = torch.zeros(num_labels, 1).long()

        for i in range(num_labels):
            random_labels[i][0] = random.randint(0, self.output_dim-1)

        return self._one_hot_encode(random_labels)


class Augmentations(object):

    Counter = 0

    def __init__(self, aug_config):
        assert hasattr(self, 'name')
        self.name = 'Aug{:02d}_{}'.format(Augmentations.Counter, self.name)
        Augmentations.Counter += 1

        self.config = aug_config

        self.summary = {
            'name' : self.name,
            }

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def augment(self, states, actions, batch=False):
        # Some preprocessing
        if not batch:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)

        assert states.size(0) == actions.size(0)

        # Apply augmentation
        augmented_states = []
        augmented_actions = []
        for i in range(states.size(0)):
            augmented_state, augmented_action = self.augment_func(states[i], actions[i])
            assert isinstance(augmented_state, torch.Tensor)
            assert isinstance(augmented_action, torch.Tensor)
            augmented_states.append(augmented_state.unsqueeze(0))
            augmented_actions.append(augmented_action.unsqueeze(0))
        augmented_states = torch.cat(augmented_states, dim=0)
        augmented_actions = torch.cat(augmented_actions, dim=0)

        return torch.squeeze(augmented_states), torch.squeeze(augmented_actions)

    def augment_func(self, states, actions):
        raise NotImplementedError


