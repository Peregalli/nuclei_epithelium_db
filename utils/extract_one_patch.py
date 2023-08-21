import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

WSI_path = '/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs'
TIFF_path = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium.tiff'

level = 0
x = 12000
y = 24000
w = 1024
h = 1024

# Run importer and get data
mask_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(TIFF_path)\
    .runAndGetOutputData()

segmentationRenderer = fast.SegmentationRenderer.create()\
    .connect(mask_pyramid_image)

fast.SimpleWindow2D.create()\
    .connect(segmentationRenderer)\
    .run()

# Extract specific patch at highest resolution
access_mask = mask_pyramid_image.getAccess(fast.ACCESS_READ)
mask = access_mask.getPatchAsImage(level, x, y, w, h)

# Run importer and get data
wsi_pyramid_image = fast.fast.TIFFImagePyramidImporter\
    .create(WSI_path)\
    .runAndGetOutputData()

segmentationRenderer = fast.SegmentationRenderer.create()\
    .connect(wsi_pyramid_image)

fast.SimpleWindow2D.create()\
    .connect(segmentationRenderer)\
    .run()

# Extract specific patch at highest resolution
access_wsi_image = wsi_pyramid_image.getAccess(fast.ACCESS_READ)
wsi_image = access_wsi_image.getPatchAsImage(level, x, y, w, h)

# Convert FAST image to numpy ndarray and plot
wsi_image = np.asarray(wsi_image)
mask = np.asarray(mask)
# This should print: (1024, 1024, 3) uint8 255 1

plt.imshow((mask*255).astype(np.uint8),alpha = 0.8)
plt.imshow(wsi_image,alpha = 0.2)
plt.savefig('image.png')
plt.show()

cv.imwrite('Lady_path.png',image[:,:,::-1])


