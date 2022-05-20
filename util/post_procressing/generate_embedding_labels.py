import os
import pickle as pk

from util.visualization.cluster_num_visualization import *

def generate_clusters(data, n_clusters,method:str, model=None):
    if model is None:
        if method =='kmeans':
            model=KMeans(n_clusters=n_clusters,max_iter=500,verbose=1)
            model.fit(data)
        elif method=='gmm':
            model=GMM(n_components=n_clusters,verbose=1)
            model.fit(data)
        else:
            raise Exception("method not implemented!")

    clusters=model.predict(data)
    return clusters,model

if __name__=='__main_':
    reconstructed_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed'
    cluster_model_path= '/util/datasets/Schwartz_mouse_v1/cluster_models'
    folder_name='3D_False_all'
    embeddings_name='embeddings_all.npy'
    bodyparts_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/data/3D_False_bodyparts.pk'
    file_path=os.path.join(reconstructed_path,folder_name,embeddings_name)

    embeddings=np.load(file_path)

    n_clusters=15
    method='gmm' # gmm or kmeans

    clusters,model=generate_clusters(embeddings,n_clusters,method,model=None)
    save_name='clusters_'+str(n_clusters)+'_'+method+'_'+embeddings_name
    save_path=os.path.join(reconstructed_path,folder_name,save_name)
    print('saving clusters data')
    np.save(save_path,clusters)

    folder_path=os.path.join(cluster_model_path,folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    save_name=os.path.join(folder_path,save_name[:-3]+'pk')

    #with open(save_name,'wb') as f:
    #    pk.dump(model,f)

if __name__=='__main__':
    method = 'gmm'
    reconstructed_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed'
    cluster_model_path = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/cluster_models/3D_False_all/clusters_15_{method}_embeddings_all.pk'
    session_ids=np.arange(0,24)
    for idx in session_ids:
        folder_name = f'3D_False_idx_{idx}_test'
        embeddings_name = 'embeddings_all.npy'
        file_path = os.path.join(reconstructed_path, folder_name, embeddings_name)

        embeddings = np.load(file_path)
        with open(cluster_model_path,'rb') as f:
            model=pk.load(f)

        if method=='gmm':
            n_clusters=model.n_components
        else:
            n_clusters=model.n_clusters


        clusters, model = generate_clusters(embeddings, n_clusters, method, model=model)
        save_name = 'clusters_' + str(n_clusters) + '_' + method + '_' + embeddings_name
        save_path = os.path.join(reconstructed_path, folder_name, save_name)
        print(f'saving clusters data of session index {idx}')
        np.save(save_path,clusters)

