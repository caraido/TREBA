import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy
import seaborn as sns
import pandas as pd

model_path='/home/roton2/PycharmProjects/TREBA/saved/Schwartz_mouse'
recon_path='/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed'

def get_cmap(n, name='brg'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

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

def draw_recon_error(ax, original:np.ndarray, recon:np.ndarray,label_name=None,extra_original=None, extra_recon=None):
    # original and recon: seq_len x seq x bodyparts
    # extra_original and extra_recon can be None, one(ndarray) or a lot of other(in the form of dict)
    #
    assert original.shape==recon.shape
    the_shape=original.shape
    original=original.reshape(the_shape[0],the_shape[1],-1,2)
    recon = recon.reshape(the_shape[0], the_shape[1], -1, 2)

    error=np.linalg.norm(original-recon,axis=-1)
    error_sum=np.sum(error,axis=-1)
    error_sum = [x for x in error_sum.transpose()]


    if extra_original is not None and extra_recon is not None:
        if isinstance(extra_original,dict) and isinstance(extra_recon,dict):

            assert len(extra_original)==len(extra_recon)
            #assert all([the_shape==item.shape for item in extra_original.values()])
            #assert all([the_shape==item.shape for item in extra_recon.values()])
            assert all([name in list(extra_recon.keys()) for name in extra_original.keys()])

            spacing=2*len(extra_original)
            colors=get_cmap(len(extra_original)+1)
            bx=[]


            bx.append(ax.boxplot(error_sum, positions=np.array(range(len(error_sum)))*spacing-1.2,sym='',widths=0.6))
            set_box_color(bx[0],colors(0))
            ax.plot([],c=colors(0),label=label_name)

            for idx,(name, item_original) in enumerate(extra_original.items()):
                item_recon=extra_recon[name]
                item_original=item_original.reshape(item_original.shape[0],item_original.shape[1],-1,2)
                item_recon = item_recon.reshape(item_recon.shape[0], item_recon.shape[1], -1, 2)

                error_item=np.linalg.norm(item_original-item_recon,axis=-1)
                error_item_sum=np.sum(error_item,axis=-1)
                error_item_sum=[x for x in error_item_sum.transpose()]
                bx.append(ax.boxplot(error_item_sum, positions=np.array(range(len(error_item_sum)))*spacing+1.2*idx,sym='',widths=0.6))
                set_box_color(bx[idx+1],colors(idx+1))
                ax.plot([],c=colors(idx+1),label=name)
            ax.legend()
            ax.set_xticks(list(range(0,len(error_sum)*spacing,spacing)),[str(x+1) for x in range(len(error_sum))])
            ax.set_xlabel('frame Num in a sequence.')
            ax.set_ylabel('reconstrunction loss')

        if isinstance(extra_original,np.ndarray) and isinstance(extra_recon,np.ndarray):
            assert extra_original.shape==extra_recon.shape
            the_shape=extra_original.shape
            extra_original = extra_original.reshape(the_shape[0], the_shape[1], -1, 2)
            extra_recon = extra_recon.reshape(the_shape[0], the_shape[1], -1, 2)

            error_extra=np.linalg.norm(extra_original-extra_recon,axis=-1)
            error_extra_sum=np.sum(error_extra,axis=-1)
            error_extra_sum=[x for x in error_extra_sum.transpose()]
            bx1=ax.boxplot(error_sum,positions=np.array(range(len(error_sum)))*2.0-0.4, sym='', widths=0.3)
            bx2=ax.boxplot(error_extra_sum, positions=np.array(range(len(error_sum))) * 2.0 + 0.4, sym='', widths=0.3)
            set_box_color(bx1, '#D7191C')  # colors are from http://colorbrewer2.org/
            set_box_color(bx2, '#2C7BB6')
            # draw temporary red and blue lines and use them to create a legend
            ax.plot([], c='#D7191C', label='Training set')
            ax.plot([], c='#2C7BB6', label='Testing set')
            ax.legend()
            ax.set_xticks(list(range(0, len(error_sum) * 2, 2)),[str(x+1) for x in range(len(error_sum))])
            ax.set_xlabel('frame Num in a sequence')
            ax.set_ylabel('reconstruction loss: MSE')
    else:

        ax.boxplot(error_sum,sym='')
        ax.set_xlabel('frame index in a sequence.')
        ax.set_ylabel('reconstruction loss')


if __name__=='__main__':
    log_path=os.path.join(model_path,'run_8','log.json')
    original_train_path=r'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_test/original_train.npy'
    recon_train_path=r'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_test/reconstructed_train.npy'
    original_test_path=r'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_test/original_test.npy'
    recon_test_path=r'/home/roton2/PycharmProjects/TREBA/util/datasets/Schwartz_mouse_v1/reconstructed/3D_False_test/reconstructed_test.npy'
    original_train=np.load(original_train_path)
    original_test=np.load(original_test_path)
    recon_train=np.load(recon_train_path)
    recon_test=np.load(recon_test_path)


    with open(log_path,'r') as f:
        log=json.load(f)

    all_loss_term=log[-1]['train']['losses']

    training_loss = [x['train']['losses'][y] for y in all_loss_term.keys() for x in log]
    training_loss = list(np.sum(np.array(training_loss).reshape(-1, len(log)), axis=0))
    val_loss = [x['test']['losses'][y] for y in all_loss_term.keys() for x in log]
    val_loss = list(np.sum(np.array(val_loss).reshape(-1, len(log)), axis=0))
    kernel_size = 20
    kernel = np.ones(kernel_size) / kernel_size
    training_loss=np.convolve(training_loss, kernel, mode='valid')
    val_loss = np.convolve(val_loss, kernel, mode='valid')

    fig=plt.figure()
    ax1=fig.add_subplot(211)
    draw_train_val_loss(ax1,training_loss,val_loss)
    ax1.set_ylim([-50,1900])
    ax2=fig.add_subplot(212)
    draw_recon_error(ax2,extra_original=original_train,extra_recon=recon_train,original=original_test,recon=recon_test,label_name=None)
    plt.tight_layout()
    plt.show()