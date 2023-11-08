import sys

sys.path.append('/home/sofia/Documents/FING/Proyecto/clasificacion_de_nucleos/nuclei_epithelium_db/src')

import os
import cv2
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from posprocesing.feature_extraction import nuclei_feature_extraction

# Minimum and maximum values of features from training data
min_values = np.array([ 0.00000000e+00, -1.00000000e+00,  8.56904367e+00,  0.00000000e+00, 6.45690849e-03,  0.00000000e+00])
max_values = np.array([2.47172100e+03, 6.01276822e-01, 2.34000000e+02, 6.76460977e-02, 1.00000000e+00, 7.46256951e+00])


# Dictionary with BGR colors
colors = {
    0: (0, 0, 255),        # Red
    1: (0, 165, 255),      # Orange
    2: (255, 0, 0),        # Blue
    3: (0, 255, 255),      # Yellow
    4: (255, 0, 255),      # Purple
    5: (255, 255, 0),      # Cyan
    6: (255, 0, 255),      # Magenta
    7: (0, 255, 0),        # Green
    8: (192, 192, 192),    # Silver
    9: (147, 20, 255),     # Pink
    10: (128, 128, 0),     # Teal
    11: (0, 42, 42),       # Brown
    12: (128, 128, 128),   # Gray
    13: (0, 128, 128),     # Olive
    14: (0, 0, 128)        # Maroon
}

# Function for making inference with kmeans in a subset of 3 images and plotting the results
def run_kmeans_test(images: list, percentile_lower: np.ndarray, percentile_upper: np.ndarray, nuclei_df: pd.DataFrame, \
                    images_names: np.ndarray, colors: dict, model):
    '''
    cv2 documentation about functions use in this script:
    
    cv2.drawContours parameters:
        img -> source image
        contours -> contours which should be passed as a Python list
        index of contours -> useful when drawing individual contour. To draw all contours, pass -1.
        color
        thickness
    '''

    fig = plt.figure(figsize=(15,15))

    for i in np.arange(0,5,2):

        image_copy = images[i%3].copy()

        # Normalize each feature using minimum and maximum values extracted from training data
        X_test = nuclei_df[nuclei_df['image_name']==images_names[i%3]][['area', 'circularity', 'intensity', 'texture_R', 'texture_U', 'texture_E']]
        X_test = (X_test - percentile_lower) / (percentile_upper - percentile_lower)

        # Run the model over the selected images
        prediction = model.predict(X_test.values)

        # Mark each cluster with a different color
        for label in np.unique(model.labels_):
            contours_list = nuclei_df[nuclei_df['image_name']==images_names[i%3]][prediction==label]['contour'].to_list()
            cv2.drawContours(image_copy, contours_list, -1, colors[label], 3)
        
        fig.add_subplot(3, 2, i+1)
        plt.imshow(images[i%3])
        plt.axis('off')
        fig.add_subplot(3, 2, i+2)
        plt.imshow(image_copy)
        plt.axis('off')


    fig.suptitle('Kmeans results', fontsize=25)
    plt.tight_layout()
    plt.show() 



parser = argparse.ArgumentParser(description='Kmeans inference')
parser.add_argument('-s', '--src_folder', help="Root to WSI patches and posprocess mask patches. Both must be in the same folder.", type=str)
parser.add_argument('-m', '--model_path', help="Path to kmeans model.", type=str)


def main():
    args = parser.parse_args()

    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)

    PATH_TO_MASKS = os.path.join(args.src_folder,'new_masks')
    PATH_TO_IMAGE = os.path.join(args.src_folder,'patches')
    
    images_name = os.listdir(PATH_TO_IMAGE)

    # Load model
    kmeans_path = args.model_path
    model = joblib.load(os.path.join(kmeans_path,'kmeans_5.joblib'))

    # Create df to save nuclei data
    nuclei_df = pd.DataFrame(columns=['image_name', 'contour', 'perimeter', 'area', 'circularity', 'intensity', \
                                      'texture_R', 'texture_U', 'texture_E'])

    
    for image_name in tqdm(images_name[:100]):
        # Load image and mask
        image = cv2.imread(os.path.join(PATH_TO_IMAGE, image_name))
        mask = cv2.imread(os.path.join(PATH_TO_MASKS, image_name))
        mask = mask[:,:,config['channel_nuclei']]

        # Extract nuclei features and save them in the df
        nuclei_data_dict = nuclei_feature_extraction(image, mask)
        nuclei_data_dict['image_name'] = image_name
        tmp_df = pd.DataFrame(nuclei_data_dict)
        nuclei_df = pd.concat([nuclei_df, tmp_df], ignore_index=True)

    # Select 3 random images
    selected_images = np.random.choice(np.unique(nuclei_df['image_name']), 3, replace=False)
    # Load images
    images = [cv2.imread(os.path.join(PATH_TO_IMAGE, image_name)) for image_name in selected_images]
    # Run kmeans and plot results
    run_kmeans_test(images=images, percentile_lower= min_values, percentile_upper= max_values, nuclei_df=nuclei_df, \
                    images_names=selected_images, colors= colors, model=model)
    

if __name__ == "__main__":
    main()
