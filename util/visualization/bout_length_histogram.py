import numpy as np
import matplotlib.pyplot as plt
import os

def plot_hist(data_dict,name):
    # this plot a bar plot frequency of each clusters
    # plus the histogram of bout length of all clusters
    fig=plt.figure()
    ax1=fig.add_subplot(2,1,1)
    frequency=[a['bouts'] for a in data_dict.values()]
    cluster_name=list(data_dict.keys())
    cluster_name=[str(a) for a in cluster_name]
    ax1.bar(cluster_name,frequency)

    ax2=fig.add_subplot(2,1,2)
    bout_length=[]
    for i in data_dict.values():
        bout_length.append(i['bout_length'])
    bout_length=[item for sublist in bout_length for item in sublist]
    ax2.hist(bout_length,bins=20000,density=True,cumulative=True,histtype='step')
    ax2.set_xscale('log')
    ax2.set_xlim([1,1000])
    plt.title(name)
    plt.tight_layout()
    plt.show()



if __name__=='__main__':
    # load cluster data
    # two types of data organizations: many npys under different folders/ one single file
    mode=0 # 0 means many npys under different folders. 1 means one single file
    root_folder = r'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed'
    save_folder= r''
    folder_name='clusters_all.npy'
    if mode:
        path=os.path.join(root_folder,'3D_False_all',folder_name)
        data=np.load(path)
    else:
        items=os.listdir(root_folder)
        data=[]
        for i in items:
            path=os.path.join(root_folder,i,folder_name)
            single_data=np.load(path)
            data.append(list(single_data))
        data=np.array([item for sublist in data for item in sublist])

    categories=dict.fromkeys(np.unique(data),None)
    for key in categories.keys():
        z=np.where(data==key)[0]
        z_diff=np.diff(z)
        bouts=[]
        bout_length=1
        for i in z_diff:
            if i==1:
                bout_length+=1
            else:
                bouts.append(bout_length)
                bout_length=1
        result={'bouts':z.size,'bout_length':bouts}
        categories[key]=result

    plot_hist(categories,folder_name)

