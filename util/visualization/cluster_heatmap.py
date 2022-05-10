import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os
from util.datasets.Schwartz_mouse_v1.preprocess import alignment
from trackings_embeddings_labels import single_alignment_2D
import matplotlib.pylab as pl
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import scipy.io as sio

def find_stim_win(mat_file,sess_id):
    sessions=mat_file['sessions']
    stims=mat_file['stims']
    one_sess=sessions[sess_id]
    event_id=one_sess[0][0][0][0]
    search=[True if s[0][0][0][0]==event_id else False for s in stims]
    index=search.index(True)

    for id in range(index,index+3):
        assert stims[id][0][0][0]==event_id
        if str(stims[id][0][2][0])=='cagemate':
            win=stims[id][0][1][0]
            return str(win)

def kNN2DDens(xv, yv, width,height, neighbours, dim=2):
    """
    """
    # Create the tree
    tree = cKDTree(np.array([xv, yv]).T)
    # Find the closest nnmax-1 neighbors (first entry is the point itself)
    grid = np.mgrid[0:height, 0:width].T.reshape(width*height, dim)
    dists = tree.query(grid, neighbours)
    # Inverse of the sum of distances to each grid point.
    inv_sum_dists = 1. / dists[0].sum(1)

    # Reshape
    im = inv_sum_dists.reshape(width, height)
    return im

def rotate_trajectories(x, test_win,tar_win,arena_center):
    '''
    this function rotate the clusters at the arena center.
    need to note that the counterclockwise order of the window is A-B-C
    input:
    x: pixel location data, nose tip or skull base
    test_win: the window in which the test mouse locates
    tar_win: the window of which location you want to rotate to
    arena_center: the center of the arena by pixel
    output:
    rotated clusters
    '''
    xshape=x.shape
    x=x.reshape(-1,int(xshape[1]/2),2)
    # assign random angles just in order to make them rotational
    window_order={'A':0,'B':2/3*np.pi,'C':4/3*np.pi}
    angle=window_order[tar_win]-window_order[test_win]

    centered_x=x-arena_center
    rot_mat=np.array([[np.cos(angle),-np.sin(angle)],
                        [np.sin(angle),np.cos(angle)]])
    rot_x=np.einsum('ij,jkl->ikl',rot_mat,centered_x.T).T
    x=rot_x+arena_center
    return x.reshape(xshape[0],-1)


def reformat_traj(traj,bodyparts):
    max_likelihood = np.max(
        [np.max(traj.reshape(-1, traj.shape[-1])[:, i * 3 + 2]) for i in range(8)])
    ratio = 1 / max_likelihood  # assume the maximum of all the likelihoods is 100%
    session_trackings, all_t_mat, all_r_mat, bodyparts = alignment(traj, bodyparts)
    placeholder_mat = np.eye(3)
    session_trackings = [single_alignment_2D(tra, placeholder_mat, np.linalg.inv(np.dot(all_r_mat[i], all_t_mat[i]))) for
                         i, tra in enumerate(session_trackings)]
    session_trackings = np.array(session_trackings)*ratio
    return session_trackings,bodyparts

if __name__=='__main__':
    arena_center=[564, 426]# need to find an elegant way to import it from the config_alignment file
    session_id = np.concatenate([np.arange(94,102),
                                 np.arange(157,165),
                                 np.arange(227,235),
                                 np.arange(389,393),
                                 np.arange(421,425)]).tolist()
    session_id=np.arange(0,24).tolist()
    target_win = 'B'
    #session_id=94

    bodyparts_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/data/3D_False_bodyparts.pk'
    raw_data_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/data'
    image_path=r'/home/roton2/PycharmProjects/TREBA/images/example_top_video.png'

    with open(bodyparts_path, 'rb') as f:
        bodyparts = pk.load(f)

    saving_path='/home/roton2/PycharmProjects/TREBA/saved/Schwartz_mouse/figs/cluster_heatmaps'

    if not isinstance(session_id,list):
        cluster_path = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_idx_{session_id}_test/clusters_15_gmm_embeddings_all.npy'
        clusters = np.load(cluster_path)
        tracking_path = os.path.join(raw_data_path, f'3D_False_idx_{session_id}_test.npz')

        session_trackings=np.load(tracking_path)['data']
        session_trackings,new_bodyparts=reformat_traj(session_trackings,bodyparts=bodyparts)
        session_trackings=session_trackings[:,0,:] # take the first frame in the seq

    else:
        mat_file_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/SchwartzLabSocialBehavior.mat'
        mat_file=sio.loadmat(mat_file_path)

        stacked_traj=[]
        stacked_clusters=[]
        for sess in session_id:
            #stim_win = find_stim_win(mat_file, sess)
            stim_win=target_win
            cluster_path = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_idx_{sess}_test/clusters_15_gmm_embeddings_all.npy'
            clusters = np.load(cluster_path)
            tracking_path = os.path.join(raw_data_path, f'3D_False_idx_{sess}_test.npz')
            session_trackings = np.load(tracking_path)['data']
            session_trackings,new_bodyparts = reformat_traj(session_trackings, bodyparts=bodyparts)
            session_trackings=rotate_trajectories(session_trackings[:,0,:],
                                                  test_win=stim_win,
                                                  tar_win=target_win,
                                                  arena_center=arena_center)
            stacked_traj.append(session_trackings)
            stacked_clusters.append(clusters)
        session_trackings=np.concatenate(stacked_traj)
        clusters=np.concatenate(stacked_clusters)

    skull_base = session_trackings[:, [new_bodyparts.index('nose_x'), new_bodyparts.index('nose_y')]]
    n_clusters = len(np.unique(clusters))
    colors = pl.cm.rainbow(np.linspace(0, 1, n_clusters))
    color_dict = list(map(lambda x: colors[x], clusters))
    image=plt.imread(image_path)
    fig=plt.figure(figsize=[12,8])
    if isinstance(session_id,list):
        #session_id=f'all_cagement_at_{target_win}'
        session_id=f'habituation'
    fig.suptitle(f"Sessions ID: {session_id}")
    ax=fig.add_subplot(441)
    ax.imshow(image,cmap='gray')

    ax.scatter(skull_base[:,0],skull_base[:,1],c=color_dict,s=4)
    handles = [mpatches.Patch(color=colors[i], label=str(i)) for i in range(15)]
    #ax.legend(handles=handles, title='behaviors', loc=(1.02, 0))
    ax.axis("off")

    for i in range(n_clusters):
        # get the cluster index
        cluster=np.where(clusters==i)[0]
        points=skull_base[cluster,:]

        # draw the heatmap
        img=kNN2DDens(points[:,0],points[:,1],width=image.shape[0],height=image.shape[1],neighbours=200)
        ax=fig.add_subplot(4,4,i+2)
        extent=[0,image.shape[1],image.shape[0],0]
        ax.imshow(image,cmap='gray')
        ax.imshow(img,alpha=0.5,cmap=cm.jet,extent=extent)
        ax.set_title(f'Cluster {i}')
        ax.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(saving_path,f'session_id_{session_id}.png'))
    print('figure saved!')
    #plt.show()












