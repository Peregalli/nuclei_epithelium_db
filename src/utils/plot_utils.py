import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2 as cv
import pandas as pd

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
    return


def glands_visualization_relative_area(df : pd.DataFrame,config : dict, root_to_folder : str):
    
    WSI_NAME, model = os.path.basename(root_to_folder).split('_')
    PATH_TO_FOLDER = root_to_folder
    
    if model == 'epithelium':
        color = (255,0,0)
    else :
        color = (0,0,255)
        
    range_areas = np.arange(0,1.2,0.2)
    fig, axs = plt.subplots(1, len(range_areas) -1, figsize = (25, 5))
    fig.suptitle(f'Relative area per gland instance - {model} model', fontsize = 20)

    for i in range(len(range_areas) -1):
        min_v = range_areas[i]
        max_v = range_areas[i+1]
        patch_loc_counts = df[(min_v < df.relative_area)&(df.relative_area < max_v)]['patch_loc'].value_counts()
        
        if len(patch_loc_counts) != 0 :
            patch_loc = patch_loc_counts.idxmax()
            
            mask_fn = os.path.basename(root_to_folder)+'_'+str(config['level'])+'_'+patch_loc+'.png'
            PATH_TO_MASK = os.path.join(PATH_TO_FOLDER,'new_masks',mask_fn)
            PATH_TO_IMG = os.path.join(PATH_TO_FOLDER,'patches',mask_fn)
            df_patch = df[df.patch_loc == patch_loc]
            
            gland_mask = cv.imread(PATH_TO_MASK)[:,:,config['channel_epithelium']]
            img = cv.imread(PATH_TO_IMG)[:,:,::-1]
            interes_glands = np.zeros((gland_mask.shape))
            for index in df_patch[(min_v < df_patch.relative_area)&(df_patch.relative_area < max_v)].patch_gland_ind:
                interes_glands = interes_glands + (gland_mask == index)
            
            contours, hierarchy = cv.findContours((255*interes_glands).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            img = cv.drawContours(img.astype(np.uint8),contours, -1, color, 7)
        else:
            img = np.ones((config['patch_size'],config['patch_size']))
            gland_mask = img.copy()
        ax = axs[i]
        ax.imshow(img)
        ax.imshow(gland_mask> 0,alpha = 0.3,cmap = 'gray')
        ax.axis('off')
        ax.set_title(f'Relative area between {round(min_v, 1)} - {round(max_v,1)}', fontsize = 15)
    
    plt.savefig(os.path.join(PATH_TO_FOLDER, f'Relative_area_per_gland_instance_{model}.png'))
    plt.show()
    return
               
def plot_glands_histogram_comparison(df_1 : pd.DataFrame, df_2 : pd.DataFrame, wsi_name_1 : str, wsi_name_2 : str, threshold_area : float = 0.005,
                                    output_dir : str = './'):
    
    df_2 = df_2[df_2.relative_area > threshold_area]
    df_1 = df_1[df_1.relative_area > threshold_area]

    c_epith = 'red'
    c_glands = 'blue'

    plt.figure(figsize = (25,10))
    plt.hist(df_2.relative_area, bins=50, color = c_epith, alpha = 0.6, label = f'{wsi_name_1}')
    plt.hist(df_1.relative_area, bins=50, color = c_glands, alpha = 0.6, label = f'{wsi_name_2}')
    plt.yscale('log')
    plt.xticks(fontsize=15)  # Change the font size as needed
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.title(f'Relative area histogram comparison', fontsize = 25)
    plt.grid()
    
    plt.savefig(os.path.join(output_dir,f'{wsi_name_1}_vs_{wsi_name_2}_plot_glands_histogram_comparison.png'))
    plt.show()
    return

def plot_glands_histogram(df : pd.DataFrame, root_to_folder : str, threshold_area : float = 0.005):

    wsi_name, model = os.path.basename(root_to_folder).split('_')
    if model == 'epithelium':
        color = 'red'
    else :
        color = 'blue'

    df = df[df.relative_area > threshold_area]    

    plt.figure(figsize = (25,10))
    plt.hist(df.relative_area, bins=50, color = color, alpha = 0.6, label = f'{model} model')
    plt.yscale('log')
    plt.xticks(fontsize=15)  # Change the font size as needed
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.title(f'Glands area histograms for {wsi_name}.svs', fontsize = 25)
    plt.grid()
    
    plt.savefig(os.path.join(root_to_folder,f'{wsi_name}_plot_glands_histogram_model.png'))
    plt.show()
    return

def main():
    PATH_TO_FOLDER = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium'
    density_map = np.load(os.path.join(PATH_TO_FOLDER,'density_map.npy'))
    density_color_map_plot(density_map, gland_channel = 0, nuclei_channel = 1, title  = PATH_TO_FOLDER.split('/')[0], dst_folder = PATH_TO_FOLDER)

if __name__ == "__main__":
    main()