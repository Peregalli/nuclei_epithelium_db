def fill_mask_holes(mask : np.ndarray, resize_dim : int = 256,kernel_size : int = 5)-> np.ndarray:
    '''
    '''
    
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
    cv.floodFill(im_floodfill,None, (10,10), 255,flags = 8);
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