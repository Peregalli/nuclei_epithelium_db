import numpy as np
import sys
sys.path.append('src/')
import cv2
import os
import json
import joblib
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import entropy
import plotly.graph_objects as go
import argparse
from utils.plot_utils import nuclei_high_circularity, plot_results_kmeans

parser = argparse.ArgumentParser(description='Get nuclei features from masks.')
parser.add_argument('-s', '--src_folder', help="path to folder that mask and patches are saved", type=str)


class NucleiPosprocessing():
    def __init__(self,path_to_folder : str):
        self.path_to_folder = path_to_folder
        self.params = self.__get_params()
        self.selected_images = None
        self.min_area = 10
        self.min_features_values = np.array([1.09020000e+01, 5.83402388e-02, 9.21593688e+00, 4.75859961e-05, 6.43694598e-03, 2.72346790e+00])
        self.max_features_values = np.array([2.48667600e+03, 5.97835209e-01, 1.94816522e+02, 6.84880965e-02, 1.93625251e-01, 7.46806437e+00])  
        self.path_to_masks = os.path.join(path_to_folder,'new_masks')
        self.path_to_images = os.path.join(path_to_folder,'patches')

    def __get_params(self):
        assert os.path.exists(os.path.join(self.path_to_folder,'report.json')), 'report.json not found. Run extract_all_patches.py first'
        with open(os.path.join(self.path_to_folder,'report.json'), 'r') as config_file:
            params = json.load(config_file)
        return params

    def __get_texture_R(self, intensity_levels):
        norm_var = intensity_levels.var()/(255**2)
        return 1-1/(1+norm_var)

    def __get_texture_U(self, p_intensity):
        return np.sum(p_intensity[0]**2)

    def __get_texture_E(self, p_intensity):
        return entropy(p_intensity[0], base=2)

    def __get_nuclei_features(self, image: np.ndarray, mask: np.ndarray)-> dict:

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
            texture_R.append(self.__get_texture_R(intensity_levels))
            texture_U.append(self.__get_texture_U(p_intensity))
            texture_E.append(self.__get_texture_E(p_intensity))

            # 4. Save data of all nuclei in a dictionary.
            
        return {'contour':contours, 'perimeter': perimeters, 'area':areas, 'circularity':circularity, 
                'intensity':intensity, 'texture_R': texture_R, 'texture_U': texture_U, 'texture_E': texture_E}
        
    def extract_features(self):

        images_names= os.listdir(self.path_to_images)

        # Create df to save nuclei data
        nuclei_df = pd.DataFrame(columns=['image_name', 'contour', 'perimeter', 'area', 'circularity', 'intensity', \
                                      'texture_R', 'texture_U', 'texture_E'])

        # Load images and masks and extract nuclei features
        for image_name in tqdm(images_names):

             # Load image and mask
            image = cv2.imread(os.path.join(self.path_to_images, image_name))
            mask = cv2.imread(os.path.join(self.path_to_masks, image_name))
            mask = mask[:,:,self.params['channel_nuclei']]
            # Set a threshold in the mask to binarize it
            mask[mask>0]=255
            # Extract nuclei features and save them in the df
            nuclei_data_dict = self.__get_nuclei_features(image, mask)
            nuclei_data_dict['image_name'] = image_name
            tmp_df = pd.DataFrame(nuclei_data_dict)
            nuclei_df = pd.concat([nuclei_df, tmp_df], ignore_index=True)
        
        # Filter instances with area below min_area
        nuclei_df = nuclei_df[nuclei_df.area>self.min_area]
            
        self.nuclei_df = nuclei_df

    def normalize_features(self):
        '''
        Normalize each feature using minimum and maximum values extracted from training data
        '''
        X_norm = self.nuclei_df[['area', 'circularity', 'intensity', 'texture_R', 'texture_U', 'texture_E']]
        self.nuclei_df_norm = (X_norm-self.min_features_values)/(self.max_features_values-self.min_features_values)

    def plot_normalize_features_histogram(self):
        features = self.nuclei_df_norm.columns

        for i, feature in enumerate (features):
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(x=self.nuclei_df_norm.iloc[:,i].values, histnorm='probability'))
            
            fig.update_layout(
                title_text='Histogram of ' + feature, # title of plot
                xaxis_title_text=feature, # xaxis label
                yaxis_title_text='Probability', # yaxis label
            )
            fig.show()

    def plot_nuclei_with_high_circularity(self, percentile: int):

        # Select 3 images that contain nuclei with circularity greater than the percentile choosen
        candidates_name = self.nuclei_df[self.nuclei_df['circularity']>np.percentile(self.nuclei_df['circularity'],percentile)]['image_name']
        selected_images = np.random.choice(np.unique(candidates_name), 3, replace=False)

        # Load images
        images = [cv2.imread(os.path.join(self.path_to_images, image_name)) for image_name in selected_images]

        # Mark in the images the nuclei with high circularity
        nuclei_high_circularity(images, percentile, self.nuclei_df, selected_images)

        self.selected_images = selected_images
    
    def nuclei_features_pair_plot(self):
        '''
        Plot pair plot of the nuclei features
        '''
        sns.pairplot(self.nuclei_df_norm, corner=True)
    
    def plot_kmeans_inference(self, kmeans_path: str = 'models/kmeans_5.joblib'):

        model = joblib.load(kmeans_path)
        prediction = model.predict(self.nuclei_df_norm.values)

        if self.selected_images is None:
            self.selected_images = np.random.choice(np.unique(self.nuclei_df['image_name']), 3, replace=False)

        # Load images
        images = [cv2.imread(os.path.join(self.path_to_images, image_name)) for image_name in self.selected_images]
       
        plot_results_kmeans(images, self.nuclei_df, self.selected_images, prediction)


if __name__ == "__main__":

    args = parser.parse_args() 

    nuclei_posprocessing = NucleiPosprocessing(args.src_folder)
    nuclei_posprocessing.extract_features()
    nuclei_posprocessing.normalize_features()
    nuclei_posprocessing.plot_normalize_features_histogram()
    nuclei_posprocessing.plot_nuclei_with_high_circularity(percentile=90)
    nuclei_posprocessing.nuclei_features_pair_plot()
    nuclei_posprocessing.plot_kmeans_inference() 