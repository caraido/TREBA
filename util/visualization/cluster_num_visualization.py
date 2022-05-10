import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,Isomap
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from multiprocessing import Pool
import time
from sklearn.metrics import silhouette_score as slts
import seaborn as sns
import pandas as pd
import matplotlib.pylab as pl

def silhouette_score(model, data):
    labels=model.fit_predict(data)
    return slts(data, labels)

# only for kmeans
def elbow_method(model, data):
    model.fit(data)
    return model.inertia_

# only for gmm
def BIC_method(model,data):
    model.fit(data)
    return model.bic(data)

def visualize_embeddings_in_2D(ax,
                               embeddings:np.ndarray,
                               labels=None,
                               method='umap',
                               computer=None,
                               color_dict=None,
                               alpha=1,s=0.1):
    if computer is None:
        if method=='umap':
            computer=UMAP(n_components=2,n_neighbors=7,min_dist=0.05)
        elif method=='pca':
            computer=PCA(n_components=2)
        elif method=='tsne':
            computer=TSNE(n_components=2)
        elif method=='isomap':
            computer=Isomap(n_components=2)
        else:
            raise NotImplementedError("wrong dimension reduction method!")
        computer.fit(embeddings)

    # result is 2D now
    result=computer.transform(embeddings)

    # get colors
    if color_dict is None:
        n_labels=len(np.unique(labels))
        colors=pl.cm.rainbow(np.linspace(0,1,n_labels))
        color_dict=list(map(lambda x: colors[x],labels))
    else:
        assert len(color_dict)==len(result)

    ax.scatter(x=result[:,0],y=result[:,1],alpha=alpha,s=s,c=color_dict)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Dimension reduction with {method} method')

    return computer,result,colors


def visualize_embeddings_in_2D_all(embeddings:np.ndarray):
    # size of embedding : framesize * n_components
    assert len(embeddings.shape)==2

    fig=plt.figure()
    ax1=fig.add_subplot(221)
    visualize_embeddings_in_2D(ax1,embeddings,method='pca')

    ax2=fig.add_subplot(222)
    visualize_embeddings_in_2D(ax2,embeddings,method='umap')

    ax3=fig.add_subplot(223)
    visualize_embeddings_in_2D(ax3,embeddings,method='tsne')

    ax4=fig.add_subplot(224)
    visualize_embeddings_in_2D(ax4,embeddings,method='isomap')

    plt.show()

# use this model wrapper for multiprocessing tool
class model_wrapper:
    def __init__(self, data,model_type:str):
        # data: embeddings
        # metrics: a method that was defined above
        # model type: kmeans or gmm
        self.data=data
        self._metrics=None
        self.model_type=model_type

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self,method):
        self._metrics=method

    def __call__(self, n_clusters):
        if self.model_type=='kmeans':
            model=KMeans(n_clusters=n_clusters)
        elif self.model_type=='gmm':
            model=GMM(n_components=n_clusters,max_iter=500)
        else:
            raise Exception("wrong model type!")

        return self.metrics(model, self.data)

if __name__=='_main__':
    downsample_rate=0.01 # only 1% of the data will be used for visualization
    embedding_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_all/embeddings_all.npy'

    embeddings=np.load(embedding_path)
    draw=np.random.choice(len(embeddings), int(len(embeddings)*downsample_rate),replace=False)
    embeddings=embeddings[draw,:]
    # apply clustering directly on the embedding
    n_clusters=list(range(5,115,10))

    # initialization of the pool
    pool=Pool(processes=10)

    start=time.time()
    # for KMeans
    kmeans_wrapper=model_wrapper(data=embeddings, model_type="kmeans")
    kmeans_wrapper.metrics=elbow_method
    print('calculating elbow score on kmeans clustering ')
    kmeans_result_elbow=pool.map(kmeans_wrapper,n_clusters)

    kmeans_wrapper.metrics=silhouette_score
    print('calculating silhouette score on kmeans clustering ')
    kmeans_result_silhouette=pool.map(kmeans_wrapper,n_clusters)

    # for gmm
    gmm_wrapper=model_wrapper(data=embeddings,model_type='gmm')
    gmm_wrapper.metrics=silhouette_score
    print('calculating silhouette score on gmm ')
    gmm_result_silhouette=pool.map(gmm_wrapper,n_clusters)

    gmm_wrapper.metrics=BIC_method
    print('calculating bic on gmm')
    gmm_result_bic=pool.map(gmm_wrapper,n_clusters)
    end=time.time()

    print("time consumption is %.2f" %(end-start))

    # plotting
    fig=plt.figure()

    # for kmeans
    ax1=fig.add_subplot(211)
    ax1.set_title("KMeans")
    ax1.plot(n_clusters,kmeans_result_silhouette,'r')
    ax1.set_xlabel('number of clusters')
    ax1.set_ylabel('silhouette scores',color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2=ax1.twinx()
    ax2.plot(n_clusters,kmeans_result_elbow,'b')
    ax2.set_ylabel('elbow scores',color='b')
    ax2.tick_params(axis='y',labelcolor='b')

    # for gmm
    ax3=fig.add_subplot(212)
    ax3.set_title("GMM")
    ax3.plot(n_clusters,gmm_result_silhouette, 'r')
    ax3.set_xlabel('number of clusters')
    ax3.set_ylabel('silhouette scores', color='r')
    ax3.tick_params(axis='y', labelcolor='r')

    ax4 = ax3.twinx()
    ax4.plot(n_clusters,gmm_result_bic, 'b')
    ax4.set_ylabel('BIC scores', color='b')
    ax4.tick_params(axis='y', labelcolor='b')

    plt.tight_layout()
    plt.show()

