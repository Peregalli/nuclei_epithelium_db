import numpy as np
import cv2 as cv
import os
import json
import pandas as pd
from tqdm import tqdm 

class GlandsAreaPosprocessing():
    def __init__(self,nuclei_channel,epithelium_channel):
        self.nuclei_channel = nuclei_channel 
        self.epithelium_channel = epithelium_channel
        self.kernel_size = 3
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.glands_proportion = 0.0035
    
    def apply(self, mask : np.ndarray, mask_fn : str, )-> np.ndarray:


        return mask

def main():
    PATH_TO_FOLDER = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_glands'
    PATH_TO_MASKS = os.path.join(PATH_TO_FOLDER,'new_masks')
    PATH_TO_PATCHES = os.path.join(PATH_TO_FOLDER,'patches')
    
    with open(os.path.join(PATH_TO_FOLDER,'report.json'), 'r') as config_file:
        params = json.load(config_file)
    
    channel_glands = params['channel_epithelium']

    # Initialize dataframe
    df = pd.DataFrame(columns=['gland_id', 'patch_loc', 'patch_gland_ind', 'relative_area'])
    df = df.set_index('gland_id')
    gland_id = 0

    # Open mask
    for mask_fn in tqdm(os.listdir(PATH_TO_MASKS)):

        mask = cv.imread(os.path.join(PATH_TO_MASKS,mask_fn))[:,:,channel_glands]
        patch_loc = (mask_fn.split('.')[0]).split('_')[-2:]
        # Separate in instances
        num_labels, glands_instances, stats, centroids = cv.connectedComponentsWithStats(mask)
        for i in range(1,num_labels):
            relative_area = np.sum(glands_instances == i)/(params['patch_size']*params['patch_size'])
            df.loc[gland_id] = {'patch_loc': patch_loc[0]+'_'+patch_loc[1],'patch_gland_ind' : i, 'relative_area' : relative_area}
            gland_id += 1 
    
    df.to_csv(os.path.join(PATH_TO_FOLDER,'glands_report.csv'))
if __name__ == "__main__":
    main()