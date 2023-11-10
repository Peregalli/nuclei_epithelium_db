import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy

def R(intensity_levels):
    norm_var = intensity_levels.var()/(255**2)
    return 1-1/(1+norm_var)

def U(p_intensity):
    return np.sum(p_intensity[0]**2)

def E(p_intensity):
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
    

        
parser = argparse.ArgumentParser(description='Nuclei feature extraction')
parser.add_argument('-s', '--src_folder', help="Root to WSI patches and posprocess mask patches. Both must be in the same folder", type=str)


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

    # Load images and masks and extract nuclei features
    for image_name in tqdm(images_name[:10]):
        image = cv2.imread(os.path.join(PATH_TO_IMAGE, image_name))
        mask = cv2.imread(os.path.join(PATH_TO_MASKS, image_name))
        mask = mask[:,:,config['channel_nuclei']]
        nuclei_data_dict = nuclei_feature_extraction(image, mask)
        nuclei_data_dict['image_name'] = image_name
        tmp_df = pd.DataFrame(nuclei_data_dict)
        nuclei_df = pd.concat([nuclei_df, tmp_df], ignore_index=True)
        

    print('Extracting nuclei features finished.')


if __name__ == "__main__":
    main()