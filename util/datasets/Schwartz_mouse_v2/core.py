import os
import numpy as np
import torch
import gc

from util.datasets import TrajectoryDataset
from .label_functions import label_functions_list
#from .augmentations import augmentation_list
from util.logging import LogEntry
from .preprocess import *
import pickle

ROOT_DIR='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/data'
TRAIN_FILE='3D_False_train.pk'
TEST_FILE='3D_False_val.pk'
BODYPARTS='3D_False_bodyparts.pk'


class SchwartzMouseV2Dataset(TrajectoryDataset):

    name='Schwartz_mouse_v2'
    all_label_functions = label_functions_list

    # Default config
    _seq_len = 21
    _state_dim = 28
    _action_dim = 28
    _svd_computer=None
    _svd_mean=None
    _val_prop = 0.2

    normalize_data = True
    align_data=True
    compute_svd=False

    test_name=TEST_FILE

    def __init__(self,data_config):
        super().__init__(data_config)

    def convert_to_trajectories(self, vid_dict, trajectory_length=61, sliding_window=1):
        """
        Author: Andrew Ulmer

        This expects a dictionary of videos containing the pose estimates
        for each frame. Expects data for a single mouse! This will convert
        videos into trajectories of length trajectory_length using a moving
        window of size sliding_window.
        """
        for (video_name, pre_pose) in tqdm.tqdm(vid_dict.items()):

            #pre_pose = pre_pose[:,0,:,:].transpose(0,2,1).reshape(pre_pose.shape[0], -1)

            # Pads the beginning and end of the sequence with duplicate frames
            pad_vec = np.pad(pre_pose, ((trajectory_length // 2, trajectory_length - 1 - trajectory_length // 2), (0, 0)),
                             mode='edge')

            # Converts sequence into [number of sub-sequences, frames in sub-sequence, x/y alternating keypoints]
            trajectories = np.stack(
                [pad_vec[i:len(pad_vec) + i - trajectory_length + 1:sliding_window] for i in range(trajectory_length)], axis=1)

            vid_dict[video_name] = trajectories

        return vid_dict

    def preprocess_videos(self, vid_dict,bodyparts:list):
        for vid_name, vid in vid_dict.items():
            data_svd, _, _ = transform_to_svd_components(
                np.concatenate(vid, axis=0).reshape(-1,int(len(bodyparts)/2),2),
                bodyparts=bodyparts,# check this
                svd_computer = self._svd_computer,
                mean = self._svd_mean
            )
            vid_dict[vid_name] = np.stack(np.vsplit(data_svd, len(data_svd)//self.seq_len))

        return vid_dict

    def combine_context_states(self, context_dict):
        context_states = []
        for vid_name, vid in context_dict.items():
            context_states.append(vid)
        del context_dict
        gc.collect()
        context_states = np.concatenate(context_states)
        return context_states

    def states_and_actions(self, vid_dict):
        states, actions = [], []
        for vid_name, vid in vid_dict.items():
            states.append(vid), actions.append(vid[:,1:] - vid[:,:-1])
        return np.concatenate(states), np.concatenate(actions)

    def make_3_context(self, vid_dict, traj_len=15):
        context_dict = {k: None for k, _ in vid_dict.items()} # change this into keys
        for vid_name, vid in vid_dict.items():
            out = []
            for i, frame in enumerate(vid):
                front, back = [], []
                if (i - traj_len) < 0:
                    front.append([vid[0]])
                    back.append([vid[i + traj_len]])

                elif (i + traj_len) >= len(vid):
                    back.append([vid[-1]])
                    front.append([vid[i - traj_len]])

                else:
                    front.append([vid[i - traj_len]])
                    back.append([vid[i + traj_len]])

                out.append(np.concatenate((np.concatenate(front), np.expand_dims(frame, axis=0), np.concatenate(back))))

            context_dict[vid_name] = np.stack(out)

        return context_dict

    def _load_data(self):
        # Process configs
        global TRAIN_FILE
        global ROOT_DIR
        if 'normalize_data' in self.config:
            self.normalize_data = self.config['normalize_data']
        if 'mirror_data' in self.config:
            self.mirror_data=self.config['mirror_data']
        if 'compute_svd' in self.config:
            assert isinstance(self.config['compute_svd'], int)
            self.compute_svd = self.config['compute_svd']
        if 'test_name' in self.config:
            self.test_name = self.config['test_name']

        # what does this do?
        if 'filename' in self.config:
            TRAIN_FILE=self.config['filename']
        if 'root_data_directory' in self.config:
            ROOT_DIR=self.config['root_data_directory']
        if 'svd_computer_path' in self.config:
            self.svd_computer_path = f'{self.config["svd_computer_path"]}'
            self.mean_path = f'{self.config["mean_computer_path"]}'
        if 'val_prop' in self.config:
            self._val_prop = self.config['val_prop']

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

        if hasattr(self, 'svd_computer_path'):
            with open(self.svd_computer_path, 'rb') as f:
                self.svd_computer = pickle.load(f)
            with open(self.mean_path, 'rb') as f:
                self.mean = pickle.load(f)

        (self.train_states,
        self.train_actions,
        self.train_ctxt_states,
        self.train_ctxt_actions,
        self.test_states,
        self.test_actions,
        self.test_ctxt_states,
        self.test_ctxt_actions) = self._load_and_preprocess(train=True) # TODO: need to change this!

    def _load_and_preprocess(self,train:bool):
        global new_bodyparts
        path = os.path.join(ROOT_DIR, TRAIN_FILE if train else self.test_name)
        body_path=os.path.join(ROOT_DIR,BODYPARTS)
        with open(body_path,'rb') as f:
            bodyparts = pickle.load(f)

        with open(path,'rb') as f:
            data = pickle.load(f)

        print('Registering, filtering, and normalizing ...')
        data=self.convert_to_trajectories(data,trajectory_length=self.seq_len,sliding_window=1)

        # align the data to egocentric, more importantly, removing tailend

        data,_,_,new_bodyparts = alignment_wrapper(data,bodyparts,remove=['tail_end'])# TODO: do not hard code this
        # you can only mirror the data for ego centric trajectories
        if self.mirror_data:
            data=mirror_wrapper(data,new_bodyparts)
        self.bodyparts=new_bodyparts


        # Combine all videos
        data_combined= np.concatenate([vid for vid_name, vid in data.items()])

        # Compute SVD on train data and apply to train and test data
        data_svd=None
        if self.compute_svd:
            data_combined = data_combined.reshape((-1, 1, int(len(new_bodyparts)/2), 2))
            # Input [seq_num x seq_len, 1 mouse, 7 bodyparts, 2 xy]
            if train and self._svd_computer is None:
                data_svd, self._svd_computer, self._svd_mean = transform_to_svd_components(
                    data_combined, new_bodyparts,center_index=int(new_bodyparts.index('skull_base_x')/2), n_components=self.compute_svd,
                    svd_computer=self._svd_computer, mean=self._svd_mean,
                    stack_agents=True, save_svd=True)
            else:
                data_svd, _, _ = transform_to_svd_components(
                    data_combined, new_bodyparts,center_index=int(new_bodyparts.index('skull_base_x')/2), n_components=self.compute_svd,
                    svd_computer=self._svd_computer, mean=self._svd_mean,
                    stack_agents=True)
            # Output [seq_num x seq_len, 2, 4 + n_components]
            #data = data_svd.reshape((-1, seq_len, data_svd.shape[-1]))
        # calculate the label category

        # calculate the label threshold
        self._init_label_functions()
        if self.config['new_threshold']:
            for i,lf in enumerate(self.active_label_functions):
                label = lf.label_func(states=torch.from_numpy(data_svd),actions=None,true_label=None,full=True).numpy()
                percentile=[np.percentile(label,20),
                            np.percentile(label,40),
                            np.percentile(label,60),
                            np.percentile(label,80)]
                # save the labels to active lable function for running
                self.active_label_functions[i].thresholds=torch.FloatTensor(percentile)
                # save the labels in config file as well for saving purpose
                self.config['labels'][i]['thresholds']=percentile
        del data_combined,data_svd
        gc.collect()


        # Preprocess invidual videos to make context now
        print('Now preprocessing all videos ...')
        data = self.preprocess_videos(data,new_bodyparts)

        # Make context
        print('Now making context ...')
        context_data = self.make_3_context(data,traj_len=10) # TODO: do not hard code this

        # Combine and compute states and actions
        states, actions = self.states_and_actions(data)
        del data
        gc.collect()

        context_states = self.combine_context_states(context_data)
        del context_data
        gc.collect()

        # Split into train and test sets
        val_idx = int(len(states) * self._val_prop)

        val_states, train_states = states[:val_idx], states[val_idx:]
        val_ctxt_states, train_ctxt_states = context_states[:val_idx], context_states[val_idx:]

        val_actions = val_states[:, 1:] - val_states[:, :-1]
        train_actions = train_states[:, 1:] - train_states[:, :-1]

        val_ctxt_actions = val_ctxt_states[:, :, 1:, :] - val_ctxt_states[:, :, :-1, :]
        train_ctxt_actions = train_ctxt_states[:, :, 1:, :] - train_ctxt_states[:, :, :-1, :]

        # update dimensions
        self._seq_len=actions.shape[1]
        self._state_dim=states.shape[-1]
        self._action_dim=actions.shape[-1]

        return (torch.from_numpy(train_states),
               torch.from_numpy(train_actions),
               torch.from_numpy(train_ctxt_states),
               torch.from_numpy(train_ctxt_actions),
               torch.from_numpy(val_states),
               torch.from_numpy(val_actions),
               torch.from_numpy(val_ctxt_states),
               torch.from_numpy(val_ctxt_actions))










