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


# Function for marking in a image the nuclei with high circularity
def nuclei_high_circularity(images: list, percentile: int, nuclei_df: pd.DataFrame, images_names: np.ndarray):
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

        # Select the nuclei with high circularity and get their contours
        high_circularity_nuclei = nuclei_df[nuclei_df['image_name']==images_names[i%3]]['circularity']>np.percentile(nuclei_df['circularity'],percentile) 
        contours_high_circularity_nuclei = nuclei_df[nuclei_df['image_name']==images_names[i%3]][high_circularity_nuclei]['contour'].to_list()

        # Draw the contours in the image
        selected_image = images[i%3].copy()
        cv2.drawContours(selected_image, contours_high_circularity_nuclei, -1, (0,255,0), 3)

        
        fig.add_subplot(3, 2, i+1)
        plt.imshow(images[i%3])
        plt.axis('off')
        fig.add_subplot(3, 2, i+2)
        plt.imshow(selected_image)
        plt.axis('off')


    fig.suptitle('Nuclei with high circularity', fontsize=25)
    plt.tight_layout()
    plt.show() 



parser = argparse.ArgumentParser(description='Kmeans inference')
parser.add_argument('-s', '--src_folder', help="Root to WSI patches and posprocess mask patches. Both must be in the same folder.", type=str)



def main():
    args = parser.parse_args()

    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)

    PATH_TO_MASKS = os.path.join(args.src_folder,'new_masks')
    PATH_TO_IMAGE = os.path.join(args.src_folder,'patches')
    
    images_name = os.listdir(PATH_TO_IMAGE)

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

    percentile = 90
    # Mark in the images the nuclei with high circularity
    nuclei_high_circularity(images, percentile, nuclei_df, selected_images)

if __name__ == "__main__":
    main()
