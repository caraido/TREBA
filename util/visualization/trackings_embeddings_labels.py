import matplotlib.pyplot as plt
import os
import pickle as pk
import numpy as np
from matplotlib import gridspec
import matplotlib.patches as mpatches
from cluster_num_visualization import visualize_embeddings_in_2D
from util.visualization.animation import draw_skeleton, find_maxmin,handle
from util.datasets.Schwartz_mouse_v2.preprocess import alignment

def single_alignment_2D(trajectory: np.ndarray, t_matrix, r_matrix):
    # here we need to make (x,y,1) from (x,y)
    # the output will be (x*,y*), which will be a vector of 16 instead of 24.
    seq_shape = trajectory.shape
    new_trajectory = np.zeros([seq_shape[0], int(2 * seq_shape[1] / 2)])
    # TODO: double check this one
    for i in range(int(seq_shape[1] / 2)):
        joint = np.array([trajectory[:,2 * i], trajectory[:,2 * i + 1], np.ones([seq_shape[0]])])
        aligned = np.dot(r_matrix, np.dot(t_matrix, joint))
        normalized_aligned = aligned / aligned[-1]
        new_trajectory[:,2 * i:2 * i + 2] = normalized_aligned[:-1,:].T

    return new_trajectory

if __name__=='__main__':
    # get the path for the data
    raw_data_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/data'
    reconstructed_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed'
    all_embeddings_path=os.path.join(reconstructed_path,'3D_False_all','embeddings_all.npy')
    all_annot_path=os.path.join(reconstructed_path,'3D_False_all','clusters_15_kmeans_embeddings_all.npy')
    bodyparts_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/data/3D_False_bodyparts.pk'

    session_id=392
    session_embeddings_path=os.path.join(reconstructed_path,f'3D_False_idx_{session_id}_test','embeddings_all.npy')
    session_annot_path = os.path.join(reconstructed_path,f'3D_False_idx_{session_id}_test', 'clusters_15_kmeans_embeddings_all.npy')

    tracking_path= os.path.join(raw_data_path,f'3D_False_idx_{session_id}_test.npz')

    # load data
    all_embeddings=np.load(all_embeddings_path)
    all_annot=np.load(all_annot_path)

    session_embeddings=np.load(session_embeddings_path)
    session_annot=np.load(session_annot_path)

    with open(bodyparts_path, 'rb') as f:
        bodyparts = pk.load(f)

    session_trackings=np.load(tracking_path)['data']
    session_trackings,all_t_mat,all_r_mat,_=alignment(session_trackings,bodyparts)
    placeholder_mat=np.eye(3)
    session_trackings=[single_alignment_2D(tra,placeholder_mat,np.linalg.inv(np.dot(all_r_mat[i],all_t_mat[i]))) for i,tra in enumerate(session_trackings)]
    session_trackings=np.array(session_trackings)

    session_trackings=session_trackings[:,0,:] # take the first frame in the seq

    # downsample all embeddings and all annotation
    downsample_rate=0.02 # default 0.01
    draw=np.random.choice(len(all_embeddings), int(len(all_embeddings)*downsample_rate),replace=False)
    all_embeddings=all_embeddings[draw,:]
    all_annot=all_annot[draw]


    # plot the data
    fig=plt.figure(figsize=(10,6))
    spec = gridspec.GridSpec(ncols=2, nrows=2,
                             height_ratios=[3, 1])

    plt.tight_layout(w_pad=10)
    fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', handle)
    plt.ion()
    ax1=fig.add_subplot(spec[0])
    ax1.set_title(f'Session ID {session_id}')
    ax2=fig.add_subplot(spec[1])

    computer,results,color_dict=visualize_embeddings_in_2D(ax=ax2,
                               embeddings=all_embeddings,
                               labels=all_annot,
                               method='umap',
                               alpha=0.3,
                               s=2
                               )
    data_points=computer.transform(session_embeddings)
    handles =[ mpatches.Patch(color=color_dict[i], label=str(i)) for i in np.unique(session_annot)]
    ax2.legend(handles=handles,title='behaviors',loc=(1.02,0))

    ax3=fig.add_subplot(414)
    sparse = dict([(x,np.where(session_annot==x)) for x in np.unique(session_annot)])
    #assert len(sparse)==15
    lines=[]
    for i, label in sparse.items():
        line=ax3.eventplot(label,color=color_dict[i],linelength=20)
        lines.append(line)
    window_size= 10# unit: second
    video_length=len(session_annot)


    # update each frame
    # for ax1
    max_x, min_x, max_y, min_y = find_maxmin(session_trackings)
    # for x2
    min_x2 = np.minimum(np.min(data_points[:, 0]), np.min(results[:, 0]))
    min_y2 = np.minimum(np.min(data_points[:, 1]), np.min(results[:, 1]))
    max_x2 = np.maximum(np.max(data_points[:, 0]), np.max(results[:, 0]))
    max_y2 = np.maximum(np.max(data_points[:, 1]), np.max(results[:, 1]))

    for i in range(video_length):
        # update ax1
        for k in range(7):
            ax1.plot(session_trackings[i,2*k],session_trackings[i,2*k+1],'o')
        draw_skeleton(ax1,session_trackings[i])
        ax1.set_xlim([min_x - 1, max_x + 1])
        ax1.set_ylim([min_y - 1, max_y + 1])
        ax1.set_title(f'Session ID {session_id}')
        # update ax2 and trace

        trace=data_points[max(0,i-4):i]
        if len(trace):
            temp_points1=ax2.plot(trace[:,0],trace[:,1],markersize=5,c='r')
            
        temp_points2=ax2.plot(data_points[i,0], data_points[i,1],c='r',markersize=40)
        ax2.set_xlim([min_x2 - 1, max_x2 + 1])
        ax2.set_ylim([min_y2 - 1, max_y2 + 1])

        #updata ax3
        temp_points3=ax3.axvline(i,color='k',linewidth=4)
        ax3.set_xlim([i-int(window_size/4*15),i+int(window_size/4*15*3)])
        ax3.set_xlabel("Frame Number")

        plt.draw()
        plt.pause(0.04)
        ax1.cla()
        if len(trace):
            [temp_points.remove() for temp_points in temp_points1]
        [temp_points.remove() for temp_points in temp_points2]
        temp_points3.remove()





