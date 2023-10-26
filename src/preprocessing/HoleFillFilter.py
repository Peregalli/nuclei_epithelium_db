import numpy as np
import cv2 as cv
import os

class HoleFillFilter:
    def __init__(self,kernel_size : int = 5):
        self.kernel_size = kernel_size
    
    def apply(self, mask : np.ndarray)-> np.ndarray:
        """ Fill holes from epithelium segmentation: Closing operation followed by a flood fill\\ 
            algorithm Open CV implementation.

        Args:
            mask (np.ndarray): Output mask prediction
            resize_dim (int, optional): Defaults to 256.
            kernel_size (int, optional): Kernel size of closing operation Defaults to 5.

        Returns:
            np.ndarray: Fill mask
        """ 

        #Apply closing
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(self.kernel_size,self.kernel_size))
        closing = cv.morphologyEx(mask.copy(), cv.MORPH_CLOSE, kernel)

        #Padding border with zeros
        mask_padded = np.zeros((mask.shape[0]+2,mask.shape[1]+2), np.uint8)
        mask_padded[1:-1,1:-1] = closing

        #floodFill image
        im_floodfill = mask_padded.copy()

        first_background_coord = (np.argwhere(mask == 0)[0][0],np.argwhere(mask == 0)[0][1])
        cv.floodFill(im_floodfill,None, first_background_coord, 255,flags = 8);
        im_floodfill = im_floodfill[1:-1,1:-1]

        # Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = closing | im_floodfill_inv

        if len(np.unique(im_out)) != len(np.unique(mask_padded)):
            im_out = closing

        return im_out


def main():
    PATH_TO_MASKS = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady/masks'
    NEW_FOLDER = 'fill_masks_2'

    if NEW_FOLDER is None:
        #Replace mask into source folder
        DEST_FOLDER = PATH_TO_MASKS
    else :
        #Save fill hole mask in a new destination
        DEST_FOLDER = os.path.join(os.path.dirname(PATH_TO_MASKS),NEW_FOLDER)
        if not os.path.exists(DEST_FOLDER):
            os.mkdir(DEST_FOLDER)

    channel_glands = 1
    CHANNEL_NUCLEI = 2
    SIZE = 256 
    
    fn_path = os.listdir(PATH_TO_MASKS)
    epithelium_fill_filter = HoleFillFilter()

    for fn in fn_path:
        mask = cv.imread(os.path.join(PATH_TO_MASKS,fn)) 
        epithelium_mask = ((mask[:,:,channel_glands] > 120)*255).astype(np.uint8)

        #Reshape mask and thresholding
        mask_resized = ((cv.resize(epithelium_mask.copy(), (SIZE,SIZE), interpolation = cv.INTER_CUBIC) > 120)*255).astype(np.uint8)

        #Apply filter
        mask_fill = epithelium_fill_filter.apply(mask_resized)
        #Resize mask to original size
        im_out = ((cv.resize(mask_fill.copy(), (epithelium_mask.shape[0],epithelium_mask.shape[1]), interpolation= cv.INTER_CUBIC) > 180)*255).astype(np.uint8)
        mask[:,:,channel_glands] = im_out
        cv.imwrite(os.path.join(DEST_FOLDER,fn),mask)

if __name__ == "__main__":
    main()