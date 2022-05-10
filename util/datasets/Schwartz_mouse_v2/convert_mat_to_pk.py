import json

import numpy as np
import os
import scipy.io as sio
import argparse # maybe not
import random
import pickle


'''
Script for converting unlabeled videos from Schwartz lab behavior data .mat into .npz
The .npz files are stacked trajectories of length N.
there are in total 449 sessions
'''
default_input_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/SchwartzLabSocialBehavior_train.mat'
default_output_folder ='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/data/'
default_config_path = '/home/roton2/PycharmProjects/TREBA/configs/Schwartz_mouse/run_2.json'



def normalize(data:np.ndarray,bodyparts:list):
    """Scale by the median length/original length."""
    shape=data.shape
    flat_data=data.reshape(-1, shape[-1]) # last dimension is bodyparts
    nose_x_i=bodyparts.index('nose_x')
    nose_y_i=bodyparts.index('nose_y')
    tailbase_x_i=bodyparts.index('tail_base_x')
    tailbase_y_i=bodyparts.index('tail_base_y')

    body_length=np.linalg.norm(np.array([flat_data[:,nose_x_i]-flat_data[:,tailbase_x_i],
                               flat_data[:,nose_y_i]-flat_data[:,tailbase_y_i]]), axis=0)
    body_length_median=np.median(body_length)
    flat_data=flat_data/body_length_median
    new_data=flat_data.reshape(shape)
    return new_data

def interpolate(data):
    for i,bodypart in enumerate(data.T):
        nans,x=np.isnan(bodypart), lambda z:z.nonzero()[0]
        bodypart[nans]=np.interp(x(nans),x(~nans),bodypart[~nans])
        data[:,i]=bodypart
    return data

def sliding_window_stack(input_array, seq_length=100, sliding_window = 1):
    total = input_array.shape[0]

    return np.stack([input_array[i:total+i-seq_length+1:sliding_window]
        for i in range(seq_length)], axis = 1)


def stack_labels(input_dict, seq_length,sliding_window=1):
    '''
    Cut labels into array of seq_number x seq_length x length of body part for each item
    If sliding_window size == seq_length, there will be no overlaps in the saved trajectories.
    If sliding_window == 1, the number of trajectories will be equal to the number of input frames.
    '''
    if sliding_window is None:
        sliding_window=seq_length
    for key, val in input_dict.items():
        label_list=[]
        for label_value in val:
            # padding for body speed
            if key =='body_speed':
                label_value=np.append(label_value,label_value[-1])[:,np.newaxis]

            # Do edge padding.
            converted_padded = np.pad(label_value, ((seq_length // 2,
                                                    seq_length - 1 - seq_length // 2), (0, 0)), mode='edge')
            cut_label_list = sliding_window_stack(converted_padded, seq_length = seq_length,
                sliding_window = sliding_window)
            if len(label_list) == 0:
                label_list = cut_label_list
            else:
                label_list = np.concatenate([label_list, cut_label_list], axis=0)
        label_list = np.array(label_list)
        input_dict[key]=label_list

    return input_dict


def stack_pose_to_traj_list(input_pose,seq_length,sliding_window=1):
    '''
    Cut post list into array of seq_number x seq_length x length of body part
    If sliding_window size == seq_length, there will be no overlaps in the saved trajectories.
    If sliding_window == 1, the number of trajectories will be equal to the number of input frames.
    '''

    pose_list=[]
    if sliding_window is None:
        sliding_window = seq_length

    for pose_value in input_pose:
        # Do edge padding.
        converted_padded = np.pad(pose_value, ((seq_length//2,
            seq_length-1-seq_length//2), (0, 0)), mode='edge')

        cut_pose_list = sliding_window_stack(converted_padded, seq_length = seq_length,
            sliding_window = sliding_window)

        if len(pose_list) == 0:
            pose_list = cut_pose_list
        else:
            pose_list = np.concatenate([pose_list, cut_pose_list], axis=0)

    pose_list = np.array(pose_list)
    return pose_list

# for each session
def get_height(pose_3d,body_parts:list):
    assert pose_3d.shape[-1]==len(body_parts)==48 # make sure it is 3D
    height={}
    for bp in body_parts:
        if '_z' in bp and '_v_z' not in bp:
            i=body_parts.index(bp)
            height[bp]=pose_3d[:,i]
    return height

# for each session
def add_height(pose,height:dict,body_parts):
    assert pose.shape[-1]==len(body_parts)==int(3*len(height))==24 # make sure it is 2D
    new_pose=[]
    for name, value in height.items():
        xi=body_parts.index(name[:-1]+'x')
        pose_bp=np.hstack([pose[:,xi:xi+2],value[:,np.newaxis],pose[:,xi+2,np.newaxis]]) # x,y,height,likelihood
        new_pose.append(pose_bp)
    new_pose=np.array(new_pose).swapaxes(0,1).reshape(pose.shape[0],32)
    return new_pose

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default=default_input_path, required=False,
                        help='Directory to the .mat file that contains all the sessions')
    parser.add_argument('--output_path', type=str, default=default_output_folder,required=False,
                        help='Path to output .npz files (in the format for training/feature extraction from TREBA)')
    parser.add_argument('--data_split', type=int, default = 0.8, required = False,
                        help='Number of videos to split into train/val.' +
                            'Use -1 to disable, otherwise specify the percentage of train sessions among all sessions.')
    parser.add_argument('--skip_to_future',type=int,default=-1,required=False,
                        help='We will save the future pose after a few frames for the model to predict. -1 meaning not saving future pose')
    parser.add_argument('--no_shuffle', action='store_true', required = False,
                        help='whether to shuffle the trajectories before saving.')
    parser.add_argument('--threeD_pose',action='store_true',required=False,
                        help='if 3d pose is True, we add z information of the reconstruction 3D pose trajectories. Otherwise we look at 2D pose trajectories from top camera')
    # here we only generate data_all/data_train/data_test.
    # data_all is the same order with the original .mat file
    # data_train is shuffled for the training (only the sessions. not the sequences)
    # data_test is only 5 sessions, shuffled(only the sessions. no the sequences)

    args=parser.parse_args()

    # read .mat file
    file=sio.loadmat(args.input_path)
    dlc_raw=file['sessions']['dlc_raw']

    # currently we don't need to calculate features/labels
    '''
    # here they call it labels, but we know it's the features.
    # labels should be independent of the window information I think?
    # right now we use head_position_arc, body_speed, gaze_bino_outer, gaze_left_outer, gaze_right_outer,
    true_labels_dict = {'head_position_arc': list(file['sessions']['head_position_arc'][:,0]),
                        'body_speed': list(file['sessions']['body_speed'][:,0]),
                        'gaze_bino_outer': list(file['sessions']['gaze_bino_outer'][:,0]),
                        'gaze_left_outer': list(file['sessions']['gaze_left_outer'][:,0]),
                        'gaze_right_outer': list(file['sessions']['gaze_right_outer'][:,0])}
    '''

    # get the name of body parts and the list of trajectories for each session
    trajectory_dict = {}
    bodyparts_3D = [x[0][0] for x in list(dlc_raw[0, 0]['threeD_parts'][0, 0])]
    bodyparts = [x[0][0] for x in list(dlc_raw[0, 0]['top_parts'][0, 0])]
    #args.data_split=1
    if args.threeD_pose:
        # for each session, there is
        for i in range(len(dlc_raw)):
            pose_3d=dlc_raw[i,0]['threeD'][0,0]
            assert np.sum(np.isnan(pose_3d))==0 # assume there is no NaN in 3D pose

            pose = dlc_raw[i, 0]['top'][0, 0]
            pose=interpolate(pose)
            pose = normalize(pose,bodyparts) # session_len x 24

            height=get_height(pose_3d,bodyparts_3D)
            pose=add_height(pose,height,bodyparts) # session_len x 32

            trajectory_dict[str(i)]=pose # index with session id
    else:
        for i in range(len(dlc_raw)):
            pose=dlc_raw[i,0]['top'][0,0]
            # only 2D data requires interpolation (personal assertion)
            pose=interpolate(pose)
            pose = normalize(pose,bodyparts)
            trajectory_dict[i]=pose

    # shuffle the session
    # we need to remember the order of shuffle! --by saving it in shuffled_index
    trajectory_list = list(trajectory_dict.items())
    if not args.no_shuffle:
        random.shuffle(trajectory_list)


    if 0 < args.data_split <= 1:
        train_set_size=int(args.data_split*len(trajectory_list))

        # segment the trajactories into sequences
        train_set=dict(trajectory_list[:train_set_size])
        print("Saving array of size: "+str(len(train_set)))

        save_path=args.output_path + '3D_'+str(args.threeD_pose)+'_train.pk'
        with open(save_path,'wb') as f:
            pickle.dump(train_set,f)

        val_set=dict(trajectory_list[train_set_size:])
        print("Saving array of size: " + str(len(val_set)))
        save_path = args.output_path + '3D_' + str(args.threeD_pose) + '_val.pk'
        with open(save_path, 'wb') as f:
            pickle.dump(val_set, f)

    else:
        # Save all the videos together.
        all_data=dict(trajectory_list)
        print("Saving array of size: " + str(len(all_data)))

        if 'test' in args.input_path:
            save_path = args.output_path + '3D_' + str(args.threeD_pose) + '_test.pk'
        else:
            save_path = args.output_path + '3D_' + str(args.threeD_pose) + '_train.pk'
        with open(save_path, 'wb') as f:
            pickle.dump(all_data, f)





