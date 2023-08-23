import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


WSI_path = '/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs'
TIFF_path = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium.tiff'

level = 0
x = 12000
y = 24000
width = 1024
high = 1024


# Run importer and get data
wsi_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(WSI_path)\
    .runAndGetOutputData()

#segmentationRenderer = fast.SegmentationRenderer.create()\
#    .connect(wsi_pyramid_image)

#fast.SimpleWindow2D.create()\
#    .connect(segmentationRenderer)\
#    .run()

W = wsi_pyramid_image.getLevelWidth(level)
H = wsi_pyramid_image.getLevelHeight(level)


# Extract specific patch at highest resolution
access_wsi_image = wsi_pyramid_image.getAccess(fast.ACCESS_READ)




# Run importer and get data
mask_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(TIFF_path)\
    .runAndGetOutputData()

w = mask_pyramid_image.getLevelWidth(level)
h = mask_pyramid_image.getLevelHeight(level)

#segmentationRenderer = fast.SegmentationRenderer.create()\
#    .connect(mask_pyramid_image)

#fast.SimpleWindow2D.create()\
#    .connect(segmentationRenderer)\
#    .run()

# Extract specific patch at highest resolution
access_mask = mask_pyramid_image.getAccess(fast.ACCESS_READ)
x_ = int(x/W*w)
y_ = int(y/H*h)

wsi_image = access_wsi_image.getPatchAsImage(level, x, y, width, high)
mask = access_mask.getPatchAsImage(level, x_, y_, int(width/W*w), int(high/H*h))

# Convert FAST image to numpy ndarray and plot
wsi_image = np.asarray(wsi_image)
mask = np.asarray(mask)
mask = (mask*255).astype(np.uint8)
mask = cv.resize(mask,(1024,1024))

# This should print: (1024, 1024, 3) uint8 255 1

plt.imshow(mask,alpha = 0.8)
plt.imshow(wsi_image,alpha = 0.5)
plt.savefig(f'image_{level}_{x}_{y}.png')
plt.show()

cv.imwrite('Lady_path.png',wsi_image[:,:,::-1])


