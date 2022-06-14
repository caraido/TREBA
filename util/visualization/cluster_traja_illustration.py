import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from util.visualization.animation import find_maxmin,draw_skeleton
import matplotlib.pylab as pl
import os
from matplotlib.backends.backend_pdf import PdfPages

def closest_factors(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False

def is_prime(x):
    for i in range(2, int(x ** 0.5) + 1):
        if x % i == 0:
            return False
    return True

def get_row_col(number):
    while True:
        a,b=closest_factors(number)
        if abs(a-b)>3:
            number+=1
        else:
            return np.maximum(a,b),np.minimum(a,b)


# draw samples from clusters
if __name__ =='__main__':
    #folder='3D_False_all'
    #idx=94
    method='kmeans'
    ##cluster_path=f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/{folder}/clusters_15_{method}_embeddings_all.npy'
    #cluster_path=f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed/3D_False_train_{idx}/clusters_all.npy'
    #trajectories_path=f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed/3D_False_train_{idx}/original_all.npy'
    trajectories_path=f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_all/original_all.npy'
    mode=1 # 0 means many npys under different folders. 1 means one single file
    root_folder = r'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed'
    save_folder= r''
    if mode:
        path=os.path.join(root_folder,'3D_False_all','clusters_15_kmeans_embeddings_all.npy')
        clusters=np.load(path)
    else:
        items=os.listdir(root_folder)
        clusters=[]
        trajectories=[]
        for i in items:
            path=os.path.join(root_folder,i,'clusters_all.npy')
            single_data=np.load(path)
            clusters.append(single_data)

            path=os.path.join(root_folder,i,'original_all.npy')
            single_data=np.load(path)
            trajectories.append(single_data)
        clusters=np.concatenate(clusters)
        trajectories=np.concatenate(trajectories)
    save_folder='Schwartz_mouse_v1'
    save_path=f'/home/roton2/PycharmProjects/TREBA/saved/{save_folder}/figs'

    save_type=f'clusters_traj_{method}'
    save_path=os.path.join(save_path,save_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    trajectories=np.load(trajectories_path)
    #clusters=np.load(cluster_path)
    downsample=25
    alpha=0.5
    n_clusters=np.unique(clusters)

    row,col=get_row_col(downsample)

    colors = pl.cm.rainbow(np.linspace(0, 1, trajectories.shape[1]))

    chosen_trajectories=[]
    # find max and min
    max_x = max_y = 0
    min_x = min_y = 10e6
    for i,cluster in enumerate(n_clusters):
        this_cluster = np.where(clusters == cluster)
        draw = np.random.choice(this_cluster[0], downsample)
        chosen_trajectories.append(trajectories[draw])

        for x in range(len(trajectories[draw])):
            max_x0, min_x0, max_y0, min_y0 = find_maxmin(trajectories[draw])
            max_x=np.maximum(max_x0,max_x)
            max_y=np.maximum(max_y0,max_y)
            min_x=np.minimum(min_x0,min_x)
            min_y=np.minimum(min_y0,min_y)

    max_x=max_x*0.8
    max_y=max_y*0.6
    min_x=min_x*0.8
    min_y=min_y*0.6

    pp=PdfPages(os.path.join(save_path,'all_clusters.pdf'))

    # for each cluster
    for i,tra in enumerate(chosen_trajectories):

        # plot each cluster
        fig=plt.figure(i,figsize=(8,6))

        # plot each mouse trajectories
        for j in range(downsample):
            ax=fig.add_subplot(row,col,j+1)
            #plot each frame
            for k in range(trajectories.shape[1]):
                # plot each bodypart
                #for l in range(7):
                #    ax.plot(chosen_trajactories[j,k,2 * l], chosen_trajactories[j,k,2 * l + 1], marker='.',c=colors[k],alpha=alpha)
                draw_skeleton(ax,tra[j,k],color=colors[k],alpha=alpha)
                ax.axis('off')
                ax.set_xlim([min_x,max_x])
                ax.set_ylim([min_y,max_y])
        name=f'Cluster_{n_clusters[i]}'
        fig.suptitle(name)
        fig.tight_layout()
        #save_name=os.path.join(save_path,name+'.pdf')
        fig.savefig(pp,format='pdf')
        #pp.savefig()
        print(f'saved figure: {name}')
    pp.close()
    #plt.show()




