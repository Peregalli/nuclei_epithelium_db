import numpy as np
import sys
sys.path.append('src/')
import cv2 as cv
import os
import json
import pandas as pd
from tqdm import tqdm 
import argparse
from utils.plot_utils import glands_visualization_relative_area, plot_glands_histogram, plot_glands_histogram_comparison, subplot_masks

parser = argparse.ArgumentParser(description='Get glands features from masks.')
parser.add_argument('-s', '--src_folder', help="path to folder that mask and patches are saved", type=str)
parser.add_argument('-c', '--compare_folder', help="Another folder to compare with src_folder.", default= None, type=str)

class GlandsPosprocessing():
    def __init__(self,path_to_folder : str):
        self.path_to_folder = path_to_folder
        self.params = self.__get_params()
        self.min_relative_area = 0.005
        self.max_relative_area = None  
        self.path_to_masks = os.path.join(path_to_folder,'new_masks')
    
    def extract_features(self, save_mask_instances : bool = True):

        if os.path.exists(os.path.join(self.path_to_folder,'glands_report.csv')):
            self.glands_data = pd.read_csv(os.path.join(self.path_to_folder,'glands_report.csv'), index_col = 'gland_id')
            return

        # Initialize
        df = pd.DataFrame(columns=['gland_id', 'patch_loc', 'patch_gland_ind', 'relative_area'])
        df = df.set_index('gland_id')
        gland_id = 0
        channel_glands = self.params['channel_glands']

        # Open mask
        for mask_fn in tqdm(os.listdir(self.path_to_masks)):

            mask_ori = cv.imread(os.path.join(self.path_to_masks,mask_fn))
            mask = mask_ori[:,:,channel_glands]
            patch_loc = (mask_fn.split('.')[0]).split('_')[-2:]
            # Separate in instances
            num_labels, glands_instances, stats, centroids = cv.connectedComponentsWithStats(mask)
            for i in range(1,num_labels):
                relative_area = np.sum(glands_instances == i)/(self.params['patch_size']*self.params['patch_size'])
                if relative_area > self.min_relative_area:
                    df.loc[gland_id] = {'patch_loc': patch_loc[0]+'_'+patch_loc[1],'patch_gland_ind' : i, 'relative_area' : relative_area}
                    gland_id += 1 

            if save_mask_instances :
                mask_ori[:,:,channel_glands] = glands_instances
                cv.imwrite(os.path.join(self.path_to_masks,mask_fn), mask)

        df.to_csv(os.path.join(self.path_to_folder,'glands_report.csv'))        
        self.glands_data = df
    
    def plot_results(self):
        glands_visualization_relative_area(self.glands_data, self.params, self.path_to_folder)
        plot_glands_histogram(self.glands_data, self.path_to_folder, self.min_relative_area)

    def __get_params(self):
        assert os.path.exists(os.path.join(self.path_to_folder,'report.json')), 'report.json not found. Run extract_all_patches.py first'
        with open(os.path.join(self.path_to_folder,'report.json'), 'r') as config_file:
            params = json.load(config_file)
        return params
    
    def compare_with(self, glands_posprocessing_to_compare):
        wsi_name = os.path.basename(self.path_to_folder)
        wsi_name_to_compare = os.path.basename(glands_posprocessing_to_compare.path_to_folder)
        plot_glands_histogram_comparison(self.glands_data, glands_posprocessing_to_compare.glands_data, wsi_name,wsi_name_to_compare, self.min_relative_area)

    def get_glands_density_greatter_than(self, relative_area : float):
        return self.glands_data[self.glands_data['relative_area'] > relative_area]
    
    def show_glands_density_greatter_than(self, relative_area : float):
        glands_density = self.get_glands_density_greatter_than(relative_area)
        if len(glands_density) == 0:
            print(f'Glands density greatter than {relative_area} : 0')
            return
        
        elif len(glands_density) > 5:
            worst_cases = glands_density.sort_values(by=['relative_area'], ascending=False).head(5)

        else :
            worst_cases = glands_density.sort_values(by=['relative_area'], ascending=False)

            subplot_masks(worst_cases, self.params,self.params['output_dir'] , title = f'Glands density greatter than {relative_area}')
        


if __name__ == "__main__":

    args = parser.parse_args() 

    glands_posprocessing_1 = GlandsPosprocessing(args.src_folder)
    glands_posprocessing_1.extract_features()
    glands_posprocessing_1.plot_results()

    glands_posprocessing_1.show_glands_density_greatter_than(0.8)

    if args.compare_folder is not None:
        glands_posprocessing_2 = GlandsPosprocessing(args.compare_folder)
        glands_posprocessing_2.extract_features()
        glands_posprocessing_2.plot_results()

        glands_posprocessing_1.compare_with(glands_posprocessing_2)  