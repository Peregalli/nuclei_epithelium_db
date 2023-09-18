import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm


WSI_path = '/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs'
TIFF_path_1 = 'nuclei_tiffs/Lady.tiff'
TIFF_path_2 = 'colon_epithelium_tiffs/Lady.tiff'
bgremoved = True
output_dir = 'Lady'
patch_visualization = False

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir,'patches'))
    os.mkdir(os.path.join(output_dir,'masks'))


level = 0
patch_size = 1024
CHANNEL_NUCELI = 2
CHANNEL_EPITHELIUM = 1

def get_access_to_tiff(tiff_path : str, level : int) -> tuple:

    # Run importer and get data from mask
    mask_pyramid_image = fast.fast.TIFFImagePyramidImporter\
        .create(tiff_path)\
        .runAndGetOutputData()

    w = mask_pyramid_image.getLevelWidth(level)
    h = mask_pyramid_image.getLevelHeight(level)

    # Extract specific patch at highest resolution
    access_mask = mask_pyramid_image.getAccess(fast.ACCESS_READ)
    
    return access_mask, w, h

def get_mask_from_access(access: tuple, x : int, y : int, W: int, H:int, patch_size : int)-> np.ndarray:
    
    w = access[1]
    h = access[2]

    x = int(x/W*w)
    y = int(y/H*h)
    mask = np.asarray(access[0].getPatchAsImage(level, x, y, int(patch_size/W*w), int(patch_size/H*h)))
    mask = (mask*255).astype(np.uint8)
    mask = cv.resize(mask,(patch_size,patch_size))

    return mask 

# Run importer and get data
wsi_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(WSI_path)\
    .runAndGetOutputData()

W = wsi_pyramid_image.getLevelWidth(level)
H = wsi_pyramid_image.getLevelHeight(level)

access_wsi_image = wsi_pyramid_image.getAccess(fast.ACCESS_READ)

access_1 = get_access_to_tiff(tiff_path= TIFF_path_1, level= level)
access_2 = get_access_to_tiff(tiff_path= TIFF_path_2, level= level)

for y in tqdm(range(0,H-patch_size,patch_size)):
    for x in range(0,W-patch_size,patch_size):

        wsi_image = np.asarray(access_wsi_image.getPatchAsImage(level, x, y, patch_size,patch_size))
        
        if(bgremoved):
            gpatchmean = cv.cvtColor(wsi_image, cv.COLOR_BGR2GRAY).mean()
            if(gpatchmean>200 or gpatchmean<50):
                continue

        mask = np.zeros((patch_size,patch_size,3))
        mask[:,:,CHANNEL_NUCELI] = get_mask_from_access(access_1, x, y, W, H, patch_size)
        mask[:,:,CHANNEL_EPITHELIUM] = get_mask_from_access(access_2, x, y, W, H, patch_size)
            
        cv.imwrite(os.path.join(output_dir,'masks',f"{output_dir}_{level}_{y}_{x}.png"),mask)
        cv.imwrite(os.path.join(output_dir,'patches',f"{output_dir}_{level}_{y}_{x}.png"),wsi_image[:,:,::-1])
        
        if patch_visualization:
            plt.imshow(wsi_image)
            plt.imshow(mask,alpha = 0.3)
            plt.savefig(os.path.join(output_dir,f'image_{level}_{y}_{x}.png'))










# This should print: (1024, 1024, 3) uint8 255 1





