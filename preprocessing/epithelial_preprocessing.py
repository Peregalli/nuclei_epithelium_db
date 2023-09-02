import numpy as np
import cv2 as cv


def fill_mask_holes(mask : np.ndarray, resize_dim : int = 256,kernel_size : int = 5)-> np.ndarray:
    """ Fill holes from epithelium segmentation: Closing operation followed by a flood fill\\ 
        algorithm Open CV implementation.

    Args:
        mask (np.ndarray): Output mask prediction
        resize_dim (int, optional): Defaults to 256.
        kernel_size (int, optional): Kernel size of closing operation Defaults to 5.

    Returns:
        np.ndarray: _description_
    """
    
    mask_ori = mask.copy()
    dims = mask.shape[:2]
    
    #Reshape mask and thresholding
    mask_resized = ((cv.resize(mask_ori.copy(), (resize_dim,resize_dim), interpolation = cv.INTER_CUBIC) > 120)*255).astype(np.uint8)

    #Apply closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size))
    closing = cv.morphologyEx(mask_resized.copy(), cv.MORPH_CLOSE, kernel)
    
    #Padding border with zeros
    mask = np.zeros((mask_resized.shape[0]+2,mask_resized.shape[1]+2), np.uint8)
    mask[1:-1,1:-1] = closing
    
    #floodFill image
    im_floodfill = mask.copy()

    first_background_coord = (np.argwhere(mask_ori == 0)[0][0],np.argwhere(mask_ori == 0)[0][1])
    cv.floodFill(im_floodfill,None, first_background_coord, 255,flags = 8);
    im_floodfill = im_floodfill[1:-1,1:-1]
    
    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = closing | im_floodfill_inv
    
    #Resize mask to original size
    im_out = ((cv.resize(im_out.copy(), dims, interpolation= cv.INTER_CUBIC) > 0)*255).astype(np.uint8)
    closing = ((cv.resize(closing.copy(), dims, interpolation= cv.INTER_CUBIC) > 0)*255).astype(np.uint8)
    
    if len(np.unique(im_out)) != len(np.unique(mask)):
        im_out = closing
        
    return closing, im_out