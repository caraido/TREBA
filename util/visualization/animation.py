import json

import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import numpy as np
import pickle as pk


from tqdm import trange
import time
import sys
from util.datasets import load_dataset
import itertools
import matplotlib
matplotlib.use('TkAgg')



def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


# 2D
def draw_skeleton(ax, pose:np.ndarray,color='grey',alpha=1):
    pose=pose.reshape(-1,2) # (bodypart*xy)
    line0=[0,1]
    line1=[0,2]
    line2=[1,3]
    line3=[2,3]
    line4=[3,5]
    line5=[3,6]
    line6=[4,5]
    line7=[4,6]
    #line8=[4,7]

    collection=[line0,line1,line2,line3,line4,line5,line6,line7]#,line8]
    xx = np.hstack([pose[x,0] for x in collection])
    yy=np.hstack([pose[x,1] for x in collection])

    ax.plot(xx,yy,c=color,alpha=alpha)

def find_maxmin(trajectories):
    max_x=max_y=0
    min_x=min_y=10e6

    if trajectories.shape[-1]==16 or trajectories.shape[-1]==14:
        for i in range(int(trajectories.shape[-1]/2)):
            x=trajectories[:,i*2]
            y=trajectories[:,i*2+1]
            max_x=np.maximum(np.max(x),max_x)
            max_y=np.maximum(np.max(y),max_y)
            min_x=np.minimum(np.min(x),min_x)
            min_y = np.minimum(np.min(y), min_y)
        return max_x,min_x,max_y,min_y

def handle(event):
    if event.key == 'q':
        sys.exit(0)

# only for 2D
def compare_trajectory_animation(trajectories:np.ndarray,compare: np.ndarray,extra=0.5,train=True):
    # here the trajectories are (seq x seq_num x bodyparts)
    if compare is not None:
        assert trajectories.shape==compare.shape

    if train:
        title="training set vs reconstruction"
    else:
        title='testing set vs reconstruction'

    fig=plt.figure(figsize=(8,4))
    fig.suptitle(title)
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)


    fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', handle)
    plt.ion()

    for i,seq in enumerate(trajectories):
        max_x, min_x, max_y, min_y = find_maxmin(seq)
        max_x2, min_x2, max_y2, min_y2 = find_maxmin(compare[i])
        max_x = np.maximum(max_x, max_x2) + extra
        max_y = np.maximum(max_y, max_y2) + extra
        min_x = np.minimum(min_x, min_x2) - extra
        min_y = np.minimum(min_y, min_y2) - extra


        for j,tra in enumerate(seq):
            try:
                for k in range(7):
                    ax1.plot(tra[2*k],tra[2*k+1],'o')
                    ax2.plot(compare[i,j,2*k],compare[i,j,2*k+1],'o')
                draw_skeleton(ax1,tra)
                draw_skeleton(ax2,compare[i,j])

                ax1.set_xlim([min_x,max_x])
                ax1.set_ylim([min_y,max_y])
                ax2.set_xlim([min_x,max_x])
                ax2.set_ylim([min_y,max_y])

                ax1.set_title('original')
                ax2.set_title('reconstruction')

                plt.draw()
                plt.pause(0.1)
                ax1.cla()
                ax2.cla()
            except KeyboardInterrupt:
                plt.close('all')
                break
        plt.pause(0.3)
    plt.ioff()
    # we would like to connect nose-leftear, nose_rightear, leftear_skullbase, rightear_skullbase,
    # skullbase_righthip, skullbase_lefthip, righthip_tailbase, lefthip_tailbase,tailbase_tailtip




def stack_all_trajectory(trajectory:np.ndarray,skip=1):
    # the original trajectory should be (seq_num x seq_length x body_parts)
    body_parts_length=trajectory.shape[-1]
    return trajectory.reshape(-1,body_parts_length)

if __name__=='__main__':
    # need to note that all the testing dataset has a skip=1. A lot of overlaps
    folder_name='3D_False_test'
    reconstructed_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed'
    bodyparts_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/data/3D_False_bodyparts.pk' # temporary solution
    folder_path=os.path.join(reconstructed_path,folder_name)
    all_items=os.listdir(folder_path)

    stack=True
    train=True # will load either train or test
    #clip=[0,1000] # we focus on from frame #max(0,clip[0]) to frame #min(max(frame_num), clip[1])

    # existence check
    assert sum(['embeddings' in x for x in all_items])
    assert sum(['information' in x for x in all_items])
    assert sum(['reconstructed' in x for x in all_items])
    assert sum(['original' in x for x in all_items])

    # load bodyparts
    with open(bodyparts_path,'rb') as f:
        bodyparts=pk.load(f)

    # load information
    with open(os.path.join(folder_path,'information.json'),'r') as f:
        config=json.load(f)

    # truncate bodypart and emit likelihood
    new_bodyparts=[[bodyparts[3*i],bodyparts[3*i+1]] for i in range(int(len(bodyparts)/3))]
    new_bodyparts=list(itertools.chain(*new_bodyparts))

    if train:
        # load original trajectory
        original=np.load(os.path.join(folder_path,'original_train.npy'))

        # load reconstructed trajectory
        reconstructed=np.load(os.path.join(folder_path,'reconstructed_train.npy'))

        # load reconstructed trajectory
        embeddings = np.load(os.path.join(folder_path, 'embeddings_train.npy'))
    else:
        # load original trajectory
        original = np.load(os.path.join(folder_path, 'original_test.npy'))

        # load reconstructed trajectory
        reconstructed = np.load(os.path.join(folder_path, 'reconstructed_test.npy'))

        # load reconstructed trajectory
        embeddings = np.load(os.path.join(folder_path, 'embeddings_test.npy'))

    compare_trajectory_animation(trajectories=original,compare=reconstructed,train=train,extra=0.1)











