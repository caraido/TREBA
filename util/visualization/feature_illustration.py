import numpy as np

from feature_extraction import extract_features
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import json
import pickle as pk

if __name__=='__main__':
    type='all'
    data='3D_False_all'
    run='run_6'
    path=os.path.join('/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/features',type,data,run,'features.pk')
    with open(path,'rb') as f:
        features=pk.load(f)

    for i,(key, value) in enumerate(features.items()):
        ax=plt.subplot(3,2,i+1)
        ax.hist(value,bins=int(len(value)/50))
        ax.set_xlim([np.median(value)-np.percentile(value,70),np.median(value)+np.percentile(value,80)])
        #if i==5: # for speed xtick
        #    ax.set_xtick()
        ax.set_title(key[5:-10])

    plt.tight_layout()
    save_path=os.path.join('/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v2/features',type,data,run,'feature_histogram.pdf')
    #plt.savefig(save_path)
    plt.show()