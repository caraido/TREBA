import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pickle as pk

def categorize_feature(feature:np.ndarray):
    # name and feature need to match
    criteria=[np.percentile(feature, 30),
              np.percentile(feature,70)]
    func=np.vectorize(lambda x : 0 if x<criteria[0] else 1 if criteria[0]<x<criteria[1] else 2)
    category=func(feature)
    return category


if __name__=='__main__':
    feature_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/features'
    cluster_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed'

    type='all'
    model='run_7'
    cluster_num=15

    feature_path=os.path.join(feature_path,type)
    cluster_path=os.path.join(cluster_path,type)

    feature_items=os.listdir(feature_path)
    cluster_items=os.listdir(cluster_path)

    assert feature_items==cluster_items

    features={}
    clusters=[]
    for item in feature_items:
        feature_file=os.path.join(feature_path,item,model,'features.pk')
        cluster_file=os.path.join(cluster_path,item,model,f'clusters_{cluster_num}_gmm_embeddings_all.npy')
        cluster_file='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_all/clusters_15_gmm_embeddings_all.npy'
        save_path = os.path.join(cluster_path,item,model,f'clusters_traj_gmm_{cluster_num}','feature_confusion_mat.pdf')
        with open(feature_file,'rb') as f:
            ftr=pk.load(f)
        for key in ftr.keys():
            if key not in features.keys():
                features[key]=ftr[key]
            else:
                features[key].concatenate(ftr[key])

        clusters.append(np.load(cluster_file))

    clusters=np.array(clusters).flatten()[:-1]

    xticks = [[x[5:-9]+'low',x[5:-9]+'medium',x[5:-9]+'high'] for x in features.keys()]
    xticks=list(np.array(xticks).flatten())
    matrix_dict={}
    for key,value in features.items():
        counts=[]
        value=categorize_feature(value)
        for cluster_num in np.unique(clusters):
            idx=list(np.where(clusters==cluster_num)[0])
            marked=value[idx]
            count=[sum(marked==0)/sum(value==0)/sum(clusters==cluster_num),
                   sum(marked==1)/sum(value==1)/sum(clusters==cluster_num),
                   sum(marked==2)/sum(value==2)/sum(clusters==cluster_num)]
            counts.append(count)
        matrix_dict[key]=np.array(counts)

    matrix=np.array(list(matrix_dict.values())).swapaxes(0,1)
    matrix=matrix.reshape(len(matrix),-1)

    fig,ax=plt.subplots()
    ax.imshow(matrix.transpose())
    ax.set_yticks(np.arange(len(xticks)),labels=xticks)
    ax.set_xticks(np.unique(clusters),labels=list(np.unique(clusters)))
    ax.set_ylabel("feature name")
    ax.set_xlabel('cluster number')
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    plt.title("Confustion matrix of categorized features")
    plt.tight_layout()
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    plt.savefig(save_path)



    
    






