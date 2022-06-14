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

# apply the model on sessions/data
if __name__=='__main_':
    method = 'gmm'
    n_clusters=15
    run='run_8'
    reconstructed_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed/cagemates'
    cluster_model_path = f'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/cluster_models/3D_False_all/{run}/clusters_{n_clusters}_{method}_embeddings_all.pk'
    #session_ids=np.arange(0,24)
    items=os.listdir(reconstructed_path)
    items=['3D_False_train_231']
    for folder_name in items:
    #for idx in session_ids:
        #folder_name = f'3D_False_idx_{idx}_test'
        embeddings_name = 'embeddings_all.npy'
        file_path = os.path.join(reconstructed_path, folder_name, run,embeddings_name)

        embeddings = np.load(file_path)
        with open(cluster_model_path,'rb') as f:
            model=pk.load(f)

        if method=='gmm':
            n_clusters=model.n_components
        else:
            n_clusters=model.n_clusters


        clusters, model = generate_clusters(embeddings, n_clusters, method, model=model)
        save_name = 'clusters_' + str(n_clusters) + '_' + method + '_' + embeddings_name
        save_path = os.path.join(reconstructed_path, folder_name, run,save_name)
        #print(f'saving clusters data of session index {idx}')
        print(f'saving clusters data of session {folder_name}')
        np.save(save_path,clusters)

# generate cluster model
if __name__=='__main_':
    method = 'gmm'
    n_clusters=15
    embeddings_name='embeddings_all.npy'
    reconstructed_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed'
    model_path= '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/cluster_models'

    items=os.listdir(reconstructed_path)
    items=['3D_False_all']
    for item in items:
        embedding_path=os.path.join(reconstructed_path,item,embeddings_name)
        embeddings=np.load(embedding_path)
        save_name = 'clusters_' + str(n_clusters) + '_' + method + '_' + embeddings_name
        clusters, model = generate_clusters(embeddings, n_clusters, method)
        save_path=os.path.join(reconstructed_path, item, save_name)
        np.save(save_path, clusters)

        model_name='clusters_' + str(n_clusters) + '_' + method + '_embeddings_all.pk'
        model_save_path=os.path.join(model_path, item)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model_save_path=os.path.join(model_save_path, model_name)
        with open(model_save_path, 'wb') as f:
            pk.dump(model,f)

# generate cluster model
if __name__=='__main__':
    method = 'kmeans'
    n_clusters=15
    downsample_rate=1
    embeddings_name='embeddings_all.npy'
    reconstructed_path = '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/reconstructed/all/3D_False_all'
    model_path= '/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/cluster_models/3D_False_all'

    items=os.listdir(reconstructed_path)
    #items=['3D_False_all']
    for item in items:
        if os.path.isdir(os.path.join(reconstructed_path,item)):
            embedding_path=os.path.join(reconstructed_path,item,embeddings_name)
            embeddings=np.load(embedding_path)
            draw=np.random.choice(len(embeddings), int(len(embeddings)*downsample_rate),replace=False)
            embeddings_run=embeddings#[draw]
            save_name = 'clusters_' + str(n_clusters) + '_' + method + '_' + embeddings_name
            _, model = generate_clusters(embeddings_run, n_clusters, method)
            clusters,_=generate_clusters(embeddings,n_clusters,method,model=model)
            save_path=os.path.join(reconstructed_path, item, save_name)
            np.save(save_path, clusters)

            model_name='clusters_' + str(n_clusters) + '_' + method + '_embeddings_all.pk'
            model_save_path=os.path.join(model_path, item)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_save_path=os.path.join(model_save_path, model_name)
            with open(model_save_path, 'wb') as f:
                pk.dump(model,f)