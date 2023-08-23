import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


WSI_path = '/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs'
TIFF_path = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium.tiff'
bgremoved = True
output_dir = 'Lady'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir,'patches'))
    os.mkdir(os.path.join(output_dir,'masks'))


level = 0
#x = 12000
#y = 24000
width = 1024
high = 1024


# Run importer and get data
wsi_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(WSI_path)\
    .runAndGetOutputData()

W = wsi_pyramid_image.getLevelWidth(level)
H = wsi_pyramid_image.getLevelHeight(level)

access_wsi_image = wsi_pyramid_image.getAccess(fast.ACCESS_READ)


# Run importer and get data from mask
mask_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(TIFF_path)\
    .runAndGetOutputData()

w = mask_pyramid_image.getLevelWidth(level)
h = mask_pyramid_image.getLevelHeight(level)

# Extract specific patch at highest resolution
access_mask = mask_pyramid_image.getAccess(fast.ACCESS_READ)

for y in range(0,H-high,high):
    for x in range(0,W-width,width):

        wsi_image = np.asarray(access_wsi_image.getPatchAsImage(level, x, y, width, high))
        
        if(bgremoved):
            gpatchmean = cv.cvtColor(wsi_image, cv.COLOR_BGR2GRAY).mean()
            if(gpatchmean>200 or gpatchmean<50):
                continue
        
        x_ = int(x/W*w)
        y_ = int(y/H*h)
        mask = np.asarray(access_mask.getPatchAsImage(level, x_, y_, int(width/W*w), int(high/H*h)))
        mask = (mask*255).astype(np.uint8)
        mask = cv.resize(mask,(1024,1024))
            
        cv.imwrite(os.path.join(output_dir,'masks',f"{output_dir}_{level}_{y}_{x}.png"),mask)
        cv.imwrite(os.path.join(output_dir,'patches',f"{output_dir}_{level}_{y}_{x}.png"),wsi_image[:,:,::-1])
        
        plt.imshow(wsi_image)
        plt.imshow(mask,alpha = 0.5)
        plt.savefig(os.path.join(output_dir,f'image_{level}_{x}_{y}.png'))
        plt.show()

cv.imwrite('Lady_path.png',wsi_image[:,:,::-1])









# This should print: (1024, 1024, 3) uint8 255 1





