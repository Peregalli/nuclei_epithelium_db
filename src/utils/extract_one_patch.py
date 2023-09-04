import fast
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

WSI_path = '/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs'
TIFF_path_1 = 'nuclei_tiffs/Lady.tiff'
TIFF_path_2 = 'colon_epithelium_tiffs/Lady.tiff'

#'/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium.tiff'

level = 0
x = 12000
y = 24000
patch_size = 1024


def get_mask_patch(tiff_path: str, x: int, y : int, level : int,  W : int, H : int, patch_size : int) -> np.ndarray:
    """ Get a patch from a .tiff file scaled in order to be aligned with original WSI

    Args:
        tiff_path (str): .tiff file path
        x (int): x coordinate of original WSI
        y (int): y coordinate of origianal WSI
        level (int): FAST parameter WSI level
        W (int): WSI original width
        H (int): WSI original height
        patch_size (int): size of the original patch 

    Returns:
        _type_: boolean mask (0,255)
    """

    # Run importer and get data
    mask_pyramid_image = fast.fast.TIFFImagePyramidImporter\
        .create(tiff_path)\
        .runAndGetOutputData()

    w = mask_pyramid_image.getLevelWidth(level)
    h = mask_pyramid_image.getLevelHeight(level)

    access_mask = mask_pyramid_image.getAccess(fast.ACCESS_READ)
    
    #Transform coordinates to mask space
    x = int(x/W*w)
    y = int(y/H*h)

    mask = access_mask.getPatchAsImage(level, x, y, int(patch_size/W*w), int(patch_size/H*h))

    # Convert FAST image to numpy ndarray 
    mask = np.asarray(mask)
    mask = (mask*255).astype(np.uint8)
    mask = cv.resize(mask,(patch_size,patch_size))

    return mask


def main():
    # Run importer and get data
    wsi_pyramid_image = fast.fast.TIFFImagePyramidImporter\
        .create(WSI_path)\
        .runAndGetOutputData()

    W = wsi_pyramid_image.getLevelWidth(level)
    H = wsi_pyramid_image.getLevelHeight(level)


    # Extract specific patch at highest resolution
    access_wsi_image = wsi_pyramid_image.getAccess(fast.ACCESS_READ)
    wsi_image = access_wsi_image.getPatchAsImage(level, x, y, patch_size,patch_size)
    wsi_image = np.asarray(wsi_image)

    mask = np.zeros((patch_size,patch_size,3))
    mask[:,:,0] = get_mask_patch(TIFF_path_1, x, y, level, W, H, patch_size)
    if TIFF_path_2 is not None:
        mask[:,:,1] = get_mask_patch(TIFF_path_2, x, y, level, W, H, patch_size)

    # This should print: (1024, 1024, 3) uint8 255 1
    plt.imshow(wsi_image)
    plt.imshow(mask,alpha = 0.5)
    plt.savefig(f'image_{level}_{y}_{x}.png')
    plt.show()

    cv.imwrite('Lady_path.png',wsi_image[:,:,::-1])

    
if __name__ == "__main__":
    main()



