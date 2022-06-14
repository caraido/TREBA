import numpy as np
import torch
import math

from util.datasets import LabelFunction
from util.datasets.Schwartz_mouse_v2.preprocess import *

global mean_path
global svd_computer_path

# I put svd to keypoint code here
class LabelFunctionWrapper(LabelFunction):


    def __init__(self,lf_config,bodyparts:list):
        super().__init__(lf_config,output_dim=1)

        self.bodyparts=bodyparts
        if 'svd_computer' in lf_config:
            self.svd_computer=lf_config['svd_computer']
            self.mean=lf_config['mean']
        else:
            with open(svd_computer_path,'rb') as f:
                self.svd_computer=pickle.load(f)
            with open(mean_path,'rb') as f:
                self.mean=pickle.load(f)

class DistanceBetweenEars(LabelFunctionWrapper):

    name='distance_between_ears'

    def label_func(self, states, actions, true_label=None,full=False):
        keypoints=transform_svd_to_keypoints(states.numpy()[:-1],
                                             self.svd_computer,
                                             self.mean,stack_agents=True) # TODO: check the last parameter
        # do we need to unnormalize the keypoints?
        #keypoints=keypoints.reshape(-1,len(self.bodyparts)/2,2)
        leftear_x=keypoints[:,self.bodyparts.index('left_ear_x')]
        leftear_y=keypoints[:,self.bodyparts.index('left_ear_y')]
        rightear_x=keypoints[:,self.bodyparts.index('right_ear_x')]
        rightear_y=keypoints[:,self.bodyparts.index('right_ear_y')]

        distance=np.sqrt(np.square(leftear_x-rightear_x)+np.square(leftear_y-rightear_y))
        label_tensor=torch.from_numpy(distance)
        label_tensor=label_tensor.to(states.device)
        if full:
            return label_tensor
        else:
            return torch.mean(label_tensor.float())


class SkullBaseToTailBaseLength(LabelFunctionWrapper):
    name='skullbase_to_tailbase_length'
    def label_func(self, states, actions, true_label=None,full=False):
        keypoints=transform_svd_to_keypoints(states.numpy()[:-1],
                                             self.svd_computer,
                                             self.mean,stack_agents=True) # TODO: check the last parameter
        # do we need to unnormalize the keypoints?
        #keypoints=keypoints.reshape(-1,len(self.bodyparts)/2,2)
        skullbase_x=keypoints[:,self.bodyparts.index('skull_base_x')]
        skullbase_y=keypoints[:,self.bodyparts.index('skull_base_y')]
        tailbase_x=keypoints[:,self.bodyparts.index('tail_base_x')]
        tailbase_y=keypoints[:,self.bodyparts.index('tail_base_y')]

        distance=np.sqrt(np.square(skullbase_x-tailbase_x)+np.square(skullbase_y-tailbase_y))
        label_tensor=torch.from_numpy(distance)
        label_tensor=label_tensor.to(states.device)
        if full:
            return label_tensor
        else:
            return torch.mean(label_tensor.float())

class SkullBaseToTailBaseLengthVar(LabelFunctionWrapper):
    name='skullbase_to_tailbase_length'
    def label_func(self, states, actions, true_label=None,full=False):
        keypoints=transform_svd_to_keypoints(states.numpy()[:-1],
                                             self.svd_computer,
                                             self.mean,stack_agents=True) # TODO: check the last parameter
        # do we need to unnormalize the keypoints?
        #keypoints=keypoints.reshape(-1,len(self.bodyparts)/2,2)
        skullbase_x=keypoints[:,self.bodyparts.index('skull_base_x')]
        skullbase_y=keypoints[:,self.bodyparts.index('skull_base_y')]
        tailbase_x=keypoints[:,self.bodyparts.index('tail_base_x')]
        tailbase_y=keypoints[:,self.bodyparts.index('tail_base_y')]

        distance=np.sqrt(np.square(skullbase_x-tailbase_x)+np.square(skullbase_y-tailbase_y))
        label_tensor=torch.from_numpy(distance)
        label_tensor=label_tensor.to(states.device)
        if full:
            return label_tensor
        else:
            return torch.var(label_tensor.float())


class HeadBodyLengthRatio(LabelFunctionWrapper):
    name='head_body_ratio'
    def label_func(self, states, actions, true_label=None,full=False):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1],
                                               self.svd_computer,
                                               self.mean, stack_agents=True)  # TODO: check the last parameter
        skullbase_x=keypoints[:,self.bodyparts.index('skull_base_x')]
        skullbase_y=keypoints[:,self.bodyparts.index('skull_base_y')]
        tailbase_x=keypoints[:,self.bodyparts.index('tail_base_x')]
        tailbase_y=keypoints[:,self.bodyparts.index('tail_base_y')]
        nose_x=keypoints[:,self.bodyparts.index('nose_x')]
        nose_y = keypoints[:, self.bodyparts.index('nose_y')]

        tail_neck_distance = np.sqrt(np.square(skullbase_x - tailbase_x) + np.square(skullbase_y - tailbase_y))
        nose_neck_distance=np.sqrt(np.square(skullbase_x - nose_x) + np.square(skullbase_y - nose_y))
        ratio=nose_neck_distance/tail_neck_distance

        label_tensor = torch.from_numpy(ratio)
        label_tensor = label_tensor.to(states.device)
        if full:
            return label_tensor
        else:
            return torch.mean(label_tensor.float())


class HeadBodyAngle(LabelFunctionWrapper):
    name='head_body_angle'
    def label_func(self, states, actions, true_label=None,full=False):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1],
                                               self.svd_computer,
                                               self.mean, stack_agents=True)  # TODO: check the last parameter
        skullbase_x = keypoints[:, self.bodyparts.index('skull_base_x')]
        skullbase_y = keypoints[:, self.bodyparts.index('skull_base_y')]
        tailbase_x = keypoints[:, self.bodyparts.index('tail_base_x')]
        tailbase_y = keypoints[:, self.bodyparts.index('tail_base_y')]
        nose_x = keypoints[:, self.bodyparts.index('nose_x')]
        nose_y = keypoints[:, self.bodyparts.index('nose_y')]
        nose_vector=np.array([nose_x-skullbase_x,nose_y-skullbase_y])
        tail_vector=np.array([tailbase_x-skullbase_x,tailbase_y-skullbase_y])

        angle=angle_between(nose_vector,tail_vector)
        label_tensor = torch.from_numpy(angle)
        label_tensor = label_tensor.to(states.device)
        if full:
            return label_tensor
        else:
            return torch.mean(label_tensor.float())


class BodyHipAngle(LabelFunctionWrapper):
    name='body_hip_angle'
    def label_func(self, states, actions, true_label=None,full=False):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1],
                                               self.svd_computer,
                                               self.mean, stack_agents=True)
        skullbase_x = keypoints[:, self.bodyparts.index('skull_base_x')]
        skullbase_y = keypoints[:, self.bodyparts.index('skull_base_y')]
        tailbase_x = keypoints[:, self.bodyparts.index('tail_base_x')]
        tailbase_y = keypoints[:, self.bodyparts.index('tail_base_y')]
        lefthip_x=keypoints[:, self.bodyparts.index('left_hip_x')]
        lefthip_y = keypoints[:, self.bodyparts.index('left_hip_y')]
        righthip_x=keypoints[:, self.bodyparts.index('right_hip_x')]
        righthip_y = keypoints[:, self.bodyparts.index('right_hip_y')]

        neck_tail_vector=np.array([tailbase_x-skullbase_x,tailbase_y-skullbase_y])
        hip_vector=np.array([lefthip_x-righthip_x,lefthip_y-righthip_y])
        angle=angle_between(neck_tail_vector,hip_vector)
        label_tensor = torch.from_numpy(angle)
        label_tensor = label_tensor.to(states.device)
        if full:
            return label_tensor
        else:
            return torch.mean(label_tensor.float())


class Speed(LabelFunctionWrapper):
    name='speed'
    def label_func(self, states, actions, true_label=None,full=False):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1],
                                               self.svd_computer,
                                               self.mean, stack_agents=True)  # TODO: check the last parameter
        skullbase_x = keypoints[:, self.bodyparts.index('skull_base_x')]
        skullbase_y = keypoints[:, self.bodyparts.index('skull_base_y')]
        skullbase_x_d=np.diff(skullbase_x,axis=0)
        skullbase_y_d=np.diff(skullbase_y,axis=0)

        speed=np.sqrt(np.square(skullbase_x_d)+np.square(skullbase_y_d))
        speed=np.append(speed,speed[-1])

        label_tensor = torch.from_numpy(speed)
        label_tensor = label_tensor.to(states.device)

        if full:
            return label_tensor
        else:
            return torch.mean(label_tensor.float())