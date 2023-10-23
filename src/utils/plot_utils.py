import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def density_color_map_plot(density_map : np.ndarray, gland_channel : int, nuclei_channel : int, title : str, dst_folder : str):

    plt.figure(figsize = (30,30))
    plt.suptitle(os.path.basename(title), fontsize = 40)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])

    ax = plt.subplot(121)
    ax.set_title('Glands Densitiy Map', fontsize = 30)
    im = ax.imshow(density_map[:,:,gland_channel],cmap ='turbo', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=24)
    ax.axis('off')

    ax = plt.subplot(122)
    norm = Normalize(vmin=0, vmax= round(np.max(density_map[:,:,nuclei_channel]), 1))
    sm = ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    ax.set_title('Nuclei Densitiy Map', fontsize = 30)
    im = ax.imshow(density_map[:,:,nuclei_channel],cmap='turbo', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=24)
    ax.axis('off')
    
    plt.savefig(os.path.join(dst_folder,'density_map.png'))


def main():
    PATH_TO_FOLDER = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium'
    density_map = np.load(os.path.join(PATH_TO_FOLDER,'density_map.npy'))
    density_color_map_plot(density_map, gland_channel = 0, nuclei_channel = 1, title  = PATH_TO_FOLDER.split('/')[0], dst_folder = PATH_TO_FOLDER)

if __name__ == "__main__":
    main()