import numpy as np
import cv2 as cv
import os
from preprocessing.NucleiCleaningFilter import NucleiCleaningFilter
from preprocessing.ContourEpitheliumRemoverFilter import ContourEpitheliumRemoverFilter
import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='Preprocessing of nuclei and glands masks')
parser.add_argument('-s', '--src_folder', help="Root to patches and original mask folder. Both masks and folder must be in the same folder", type=str)
parser.add_argument('-n', '--new_folder', help="new mask destination folder ", default= 'new_masks', type=str)


def main():
    args = parser.parse_args()

    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)

    PATH_TO_MASKS = os.path.join(args.src_folder,'masks')
    PATH_TO_IMAGE = os.path.join(args.src_folder,'patches')
    NEW_FOLDER = args.new_folder

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
    contour_epithelium_filter = ContourEpitheliumRemoverFilter(kernel_size_erode = int(0.02*config['patch_size_preprocessing']), watershed_min_distance= int(0.1*config['patch_size_preprocessing']))

    for fn in tqdm(fn_path):
        mask = cv.imread(os.path.join(PATH_TO_MASKS,fn))
        image = cv.imread(os.path.join(PATH_TO_IMAGE,fn))

        #Resize image and mask
        
        image_reshaped = cv.resize(image.copy(), (config['patch_size_preprocessing'],config['patch_size_preprocessing']), interpolation = cv.INTER_CUBIC)
        mask_reshaped = ((cv.resize(mask.copy(), (config['patch_size_preprocessing'],config['patch_size_preprocessing']), interpolation = cv.INTER_CUBIC) > 120)*255).astype(np.uint8)

        #Apply filters
        mask_reshaped = nuclei_cleaning_filter.apply(mask_reshaped)
        mask_reshaped[:,:,config['channel_epithelium']] = contour_epithelium_filter.apply(image_reshaped,mask_reshaped[:,:,config['channel_epithelium']])

        new_mask = (cv.resize(mask_reshaped, (mask.shape[0],mask.shape[1]), interpolation = cv.INTER_CUBIC)).astype(np.uint8) 
        
        cv.imwrite(os.path.join(DEST_FOLDER,fn),new_mask)

    print('Preprocessing finished')

if __name__ == "__main__":
    main()