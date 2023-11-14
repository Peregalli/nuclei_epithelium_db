import numpy as np
import sys
sys.path.append('src/')
import cv2
import os
import json
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy 

import argparse
from utils.plot_utils import glands_visualization_relative_area, plot_glands_histogram, plot_glands_histogram_comparison, subplot_masks

parser = argparse.ArgumentParser(description='Get glands features from masks.')
parser.add_argument('-s', '--src_folder', help="path to folder that mask and patches are saved", type=str)
parser.add_argument('-c', '--compare_folder', help="Another folder to compare with src_folder.", default= None, type=str)

class NucleiPosprocessing():
    def __init__(self,path_to_folder : str):
        self.path_to_folder = path_to_folder
        self.params = self.__get_params()
        self.min_area = 10  
        self.path_to_masks = os.path.join(path_to_folder,'new_masks')
        self.path_to_images = os.path.join(path_to_folder,'patches')

    def __get_texture_R(intensity_levels):
        norm_var = intensity_levels.var()/(255**2)
        return 1-1/(1+norm_var)

    def __get_texture_U(p_intensity):
        return np.sum(p_intensity[0]**2)

    def __get_texture_E(p_intensity):
        return entropy(p_intensity[0], base=2)
    
    def nuclei_feature_extraction(image: np.ndarray, mask: np.ndarray)-> dict:

        '''
        cv2 documentation about functions use in this script:

        cv2.connectedComponentsWithStats outputs:
            num_labels: The total number of labels [0, ..., N-1] where 0 represents the background label.
            nuceli_instances: A grayscale image where each pixel is the label of the object at that location.
            stats: A 2D array where each row is a 5-dim vector that contains the following information, in order:
                CC_STAT_LEFT: The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
                CC_STAT_TOP: The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
                CC_STAT_WIDTH: The horizontal size of the bounding box.
                CC_STAT_HEIGHT: The vertical size of the bounding box.
                CC_STAT_AREA: The total area (in pixels) of the connected component.
            centroids: A 2D array where each row indicates the (x, y) coordinate of a centroid. The row corresponds to the label number. 

        cv2.findContours parameters:
            mode -> cv2.RETR_EXTERNAL: retrieves only the extreme outer contours.
            method -> cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments and leaves only their end points.
        '''

        # 1. Separate all nuclei instances and calculate statistics.
        num_labels, nuclei_instances, stats, centroids = cv2.connectedComponentsWithStats(mask)

        contours = []
        perimeters = []
        areas = []
        circularity = []
        intensity = []
        texture_R = []
        texture_U = []
        texture_E = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Iterate over each nucleus and calculate features.
        for j in range(1,num_labels):

            nuclei_mask = ((nuclei_instances==j)*255).astype('uint8')
            output = cv2.findContours(image=nuclei_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(output[0][0],1)
            area = cv2.contourArea(output[0][0])
            contours.append(output[0][0])
            perimeters.append(perimeter)
            areas.append(area)
            intensity_levels = gray_image[nuclei_instances==j]
            intensity.append(intensity_levels.mean())
            if (perimeter**2!=0):
                circularity.append((np.pi*4*area)/(perimeter**2))
            else:
                # append circularity value of -1 for nuclei with perimeter^2=0
                circularity.append(-1)


            # 3. Calculate the probability density function of the intensity levels in the nucleus in order to calculate different metrics regarding nucleus texture
            p_intensity = np.histogram(intensity_levels, bins=np.arange(intensity_levels.min(), intensity_levels.max()+2), density=True)
            texture_R.append(R(intensity_levels))
            texture_U.append(U(p_intensity))
            texture_E.append(E(p_intensity))

            # 4. Save data of all nuclei in a dictionary.
            
        return {'contour':contours, 'perimeter': perimeters, 'area':areas, 'circularity':circularity, 
                'intensity':intensity, 'texture_R': texture_R, 'texture_U': texture_U, 'texture_E': texture_E}
    
    
    def extract_features(self):

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
    
    def plot_results(self, hist_color : str = None):
        
        if not self.micro_meters:
            plot_glands_histogram(self.glands_data, self.path_to_folder, self.min_relative_area,color = hist_color)
            glands_visualization_relative_area(self.glands_data, self.params, self.path_to_folder)
        else:   
            pixel_scale = np.float16(self.params['pixel_scale'])/1000
            patch_size = self.params['patch_size']
            glands_data_um2 = self.glands_data.copy()
            glands_data_um2.relative_area = glands_data_um2.relative_area*((pixel_scale*patch_size)**2)

            glands_visualization_relative_area(glands_data_um2, self.params, self.path_to_folder, max_area=(pixel_scale*patch_size)**2)
            plot_glands_histogram(glands_data_um2, self.path_to_folder, threshold_area = self.min_relative_area*((pixel_scale*patch_size)**2),color = hist_color, max_value = (pixel_scale*patch_size)**2)

    def __get_params(self):
        assert os.path.exists(os.path.join(self.path_to_folder,'report.json')), 'report.json not found. Run extract_all_patches.py first'
        with open(os.path.join(self.path_to_folder,'report.json'), 'r') as config_file:
            params = json.load(config_file)
        return params
    
    def compare_with(self, glands_posprocessing_to_compare):
        wsi_name = os.path.basename(self.path_to_folder)
        wsi_name_to_compare = os.path.basename(glands_posprocessing_to_compare.path_to_folder)

        if not self.micro_meters:
            plot_glands_histogram_comparison(self.glands_data, glands_posprocessing_to_compare.glands_data, wsi_name,wsi_name_to_compare, self.min_relative_area)
        
        else :
            # in micrometers change to milimeters
            pixel_scale = np.float16(self.params['pixel_scale'])/1000
            patch_size = self.params['patch_size']
            glands_data_um2 = self.glands_data.copy()
            glands_data_um2.relative_area = glands_data_um2.relative_area*((pixel_scale*patch_size)**2)
            glands_data_to_compare_um2 = glands_posprocessing_to_compare.glands_data.copy()
            glands_data_to_compare_um2.relative_area = glands_data_to_compare_um2.relative_area*((pixel_scale*patch_size)**2)
            plot_glands_histogram_comparison(glands_data_um2, glands_data_to_compare_um2, wsi_name,wsi_name_to_compare, self.min_relative_area*((pixel_scale*patch_size)**2),
                                             max_value = (pixel_scale*patch_size)**2)

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

    glands_posprocessing_1 = GlandsPosprocessing(args.src_folder, micro_meters = True)
    glands_posprocessing_1.extract_features()
    glands_posprocessing_1.plot_results()

    glands_posprocessing_1.show_glands_density_greatter_than(0.8)

    if args.compare_folder is not None:
        glands_posprocessing_2 = GlandsPosprocessing(args.compare_folder, micro_meters = True)
        glands_posprocessing_2.extract_features()
        glands_posprocessing_2.plot_results()

        glands_posprocessing_1.compare_with(glands_posprocessing_2)  