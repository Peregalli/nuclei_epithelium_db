import numpy as np
import cv2 as cv
import os
import sys
sys.path.append('src/')
from preprocessing.NucleiCleaningFilter import NucleiCleaningFilter
from preprocessing.ContourEpitheliumRemoverFilter import ContourEpitheliumRemoverFilter
from utils.plot_utils import density_color_map_plot
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Preprocessing of nuclei and glands masks')
parser.add_argument('-s', '--src_folder', help="Root to patches and original mask folder. Both masks and folder must be in the same folder", type=str)
parser.add_argument('-n', '--new_folder', help="new mask destination folder ", default= 'new_masks', type=str)


def mask_preprocessing(src_folder : str, new_folder : str = None):

    with open(os.path.join(src_folder,'report.json'), 'r') as config_file:
        config = json.load(config_file)

    PATH_TO_MASKS = os.path.join(src_folder,'masks')
    PATH_TO_IMAGE = os.path.join(src_folder,'patches')       
    NEW_FOLDER = new_folder

    if NEW_FOLDER is None:
        #Replace mask into source folder
        DEST_FOLDER = PATH_TO_MASKS
    else :
        #Save fill hole mask in a new destination
        DEST_FOLDER = os.path.join(os.path.dirname(PATH_TO_MASKS),NEW_FOLDER)
        if not os.path.exists(DEST_FOLDER):
            os.mkdir(DEST_FOLDER)
    
    fn_path = os.listdir(PATH_TO_MASKS)

    #Define filters
    nuclei_cleaning_filter = NucleiCleaningFilter(nuclei_channel=config['channel_nuclei'], epithelium_channel= config['channel_epithelium']) 
    contour_remover_filter = ContourEpitheliumRemoverFilter(kernel_size_erode = int(0.02*config['patch_size_preprocessing']), watershed_min_distance= int(0.1*config['patch_size_preprocessing']))

    density_map = np.zeros((config['wsi_height']//config['patch_size'], config['wsi_width']//config['patch_size'],3))                       

    for fn in tqdm(fn_path):
        mask = cv.imread(os.path.join(PATH_TO_MASKS,fn))
        image = cv.imread(os.path.join(PATH_TO_IMAGE,fn))

        #Resize image and mask
        
        image_reshaped = cv.resize(image.copy(), (config['patch_size_preprocessing'],config['patch_size_preprocessing']), interpolation = cv.INTER_CUBIC)
        mask_reshaped = ((cv.resize(mask.copy(), (config['patch_size_preprocessing'],config['patch_size_preprocessing']), interpolation = cv.INTER_CUBIC) > 120)*255).astype(np.uint8)

        #Apply filters
        mask_reshaped = nuclei_cleaning_filter.apply(mask_reshaped)
        mask_reshaped[:,:,config["channel_epithelium"]] = contour_remover_filter.apply(image_reshaped,mask_reshaped[:,:,config["channel_epithelium"]])
        new_mask = (cv.resize(mask_reshaped, (mask.shape[0],mask.shape[1]), interpolation = cv.INTER_CUBIC)).astype(np.uint8) 
        
        #Save density map
        y, x = os.path.splitext(fn)[0].split('_')[-2:]
        y = int(y)
        x = int(x)

        glands_density = np.sum(new_mask[:,:,config['channel_epithelium']]>0)/(new_mask.shape[0]*new_mask.shape[1])
        nuclei_density = np.sum(new_mask[:,:,config['channel_nuclei']]>0)/(new_mask.shape[0]*new_mask.shape[1])

        density_map[y//config['patch_size'],x//config['patch_size'],0] = glands_density
        density_map[y//config['patch_size'],x//config['patch_size'],1] = nuclei_density
        density_map[y//config['patch_size'],x//config['patch_size'],2] = 1

        cv.imwrite(os.path.join(DEST_FOLDER,fn),new_mask)

    np.save(os.path.join(src_folder,'density_map'),density_map)
    density_color_map_plot(density_map, gland_channel = 0, nuclei_channel = 1, title  = os.path.split(src_folder)[-1], dst_folder = src_folder)
    print('Preprocessing finished')

if __name__ == "__main__":
    args = parser.parse_args()

    mask_preprocessing(args.src_folder, args.new_folder)