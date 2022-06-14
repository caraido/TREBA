import numpy as np
import matplotlib.pyplot as plt
from loss_visualization import draw_recon_error
import os

if __name__=='__main__':
    root_dir='/home/roton2/PycharmProjects/TREBA'
    run_6_path='util/datasets/Schwartz_mouse_v2/reconstructed/all/3D_False_all/run_6'
    run_7_path='util/datasets/Schwartz_mouse_v2/reconstructed/all/3D_False_all/run_7'
    run_8_path='util/datasets/Schwartz_mouse_v2/reconstructed/all/3D_False_all/run_8'

    original_6=np.load(os.path.join(root_dir,run_6_path,'original_all.npy'))
    recon_6 = np.load(os.path.join(root_dir,run_6_path, 'reconstructed_all.npy'))

    original_7=np.load(os.path.join(root_dir,run_7_path,'original_all.npy'))
    recon_7 = np.load(os.path.join(root_dir,run_7_path, 'reconstructed_all.npy'))

    original_8=np.load(os.path.join(root_dir,run_8_path,'original_all.npy'))
    recon_8 = np.load(os.path.join(root_dir,run_8_path, 'reconstructed_all.npy'))

    extra_original={'traj_len=21, context_len=20':original_7,'traj_len=7, context_len=4':original_8}
    extra_recon={'traj_len=21, context_len=20':recon_7,'traj_len=7, context_len=4':recon_8}

    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(111)
    draw_recon_error(ax,original_6,recon_6,
                     label_name='traj_len=21, context_len=10',
                     extra_original=extra_original,
                     extra_recon=extra_recon)

    plt.tight_layout()
    plt.show()