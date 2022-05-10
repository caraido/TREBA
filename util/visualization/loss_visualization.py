import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy
import seaborn as sns
import pandas as pd

model_path='/home/roton2/PycharmProjects/TREBA/saved/Schwartz_mouse'
recon_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed'

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def draw_train_val_loss(ax,training_loss,val_loss):
    ax.plot(training_loss)
    ax.plot(val_loss)
    ax.legend(["training loss","validation loss"])
    ax.set_xlabel('Num of epochs')
    ax.set_ylabel('sum of all losses')

def draw_recon_error(ax, original:np.ndarray, recon:np.ndarray,extra_original=None, extra_recon=None):
    # original and recon: seq_len x seq x bodyparts
    assert original.shape==recon.shape
    the_shape=original.shape
    original=original.reshape(the_shape[0],the_shape[1],-1,2)
    recon = recon.reshape(the_shape[0], the_shape[1], -1, 2)

    error=np.linalg.norm(original-recon,axis=-1)
    error_sum=np.sum(error,axis=-1)
    error_sum = [x for x in error_sum.transpose()]

    if extra_original is not None and extra_recon is not None:
        assert extra_original.shape==extra_recon.shape
        the_shape=extra_original.shape
        extra_original = extra_original.reshape(the_shape[0], the_shape[1], -1, 2)
        extra_recon = extra_recon.reshape(the_shape[0], the_shape[1], -1, 2)

        error_extra=np.linalg.norm(extra_original-extra_recon,axis=-1)
        error_extra_sum=np.sum(error_extra,axis=-1)
        error_extra_sum=[x for x in error_extra_sum.transpose()]
        bx1=ax.boxplot(error_sum,positions=np.array(range(len(error_sum)))*2.0-0.4, sym='', widths=0.6)
        bx2=ax.boxplot(error_extra_sum, positions=np.array(range(len(error_sum))) * 2.0 + 0.4, sym='', widths=0.6)
        set_box_color(bx1, '#D7191C')  # colors are from http://colorbrewer2.org/
        set_box_color(bx2, '#2C7BB6')
        # draw temporary red and blue lines and use them to create a legend
        ax.plot([], c='#D7191C', label='Training set')
        ax.plot([], c='#2C7BB6', label='Testing set')
        ax.legend()
        ax.set_xticks(list(range(0, len(error_sum) * 2, 2)),[str(x+1) for x in range(len(error_sum))])
        ax.set_xlabel('frame Num in a sequence.')
        ax.set_ylabel('reconstruction loss')
    else:

        ax.boxplot(error_sum,sym='')
        ax.set_xlabel('frame index in a sequence.')
        ax.set_ylabel('reconstruction loss')


if __name__=='__main__':
    recon_folder='3D_False_test'

    log_path=os.path.join(model_path,'run','log.json')
    with open(log_path,'r') as f:
        log=json.load(f)

    recon_folder=os.path.join(recon_path,recon_folder)

    original_train=np.load(os.path.join(recon_folder,'original_train.npy'))
    recon_train=np.load(os.path.join(recon_folder,'reconstructed_train.npy'))
    original_test=np.load(os.path.join(recon_folder,'original_test.npy'))
    recon_test=np.load(os.path.join(recon_folder,'reconstructed_test.npy'))


    training_loss=[x['train']['losses']['kl_div']+x['train']['losses']['nll']+x['train']['losses']['contrastive'] for x in log]
    val_loss=[x['test']['losses']['kl_div']+x['test']['losses']['nll']+x['test']['losses']['contrastive'] for x in log]

    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    training_loss=np.convolve(training_loss, kernel, mode='valid')
    val_loss = np.convolve(val_loss, kernel, mode='valid')

    fig=plt.figure()
    ax1=fig.add_subplot(211)
    draw_train_val_loss(ax1,training_loss,val_loss)
    ax2=fig.add_subplot(212)
    draw_recon_error(ax2,original_train,recon_train,original_test,recon_test)
    plt.tight_layout()
    plt.show()