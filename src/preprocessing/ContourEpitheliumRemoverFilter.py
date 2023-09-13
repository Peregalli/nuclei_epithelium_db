import numpy as np
import os
import cv2 as cv

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from preprocessing.HoleFillFilter import HoleFillFilter

class ContourEpitheliumRemoverFilter():
    def __init__(self, background_proportion : np.float32 = 0.15,intersection_threshold : np.float32 = 0.15, kernel_size_erode : int = 5, watershed_min_distance : int = 25):
        self.tissue_hole_fill_filter = HoleFillFilter()
        self.background_proportion = background_proportion
        self.threshold_tissue_segementatior = 230
        self.intersection_threshold = intersection_threshold
        self.kernel_size_erode = kernel_size_erode
        self.watershed_min_distance = watershed_min_distance
  
    def apply(self,image : np.ndarray, mask : np.ndarray):
        
        NO_MASK = (np.zeros(mask.shape) == mask).all()
        if NO_MASK :
            return mask
        
        #Check if image contour
        tissue_mask = self.__thresholding(image, self.threshold_tissue_segementatior)
        if self.__is_a_contour(tissue_mask, self.background_proportion):

            #Fill tissue_mask
            fill_tissue_mask = self.tissue_hole_fill_filter.apply(tissue_mask)

            #Erode tissue_mask
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(self.kernel_size_erode,self.kernel_size_erode))
            erode_mask = cv.morphologyEx(fill_tissue_mask, cv.MORPH_ERODE, kernel, iterations = 3)
        
            #Compare tissue segmentation and original epithelium mask
            contour_pixels = (erode_mask == 0 )&(mask == 255)

            #Separate epithelium 
            labels = self.__watershed_filter(mask, min_distance=self.watershed_min_distance )
            new_mask = self.__eliminate_contour_epithelium(labels,contour_pixels,self.intersection_threshold)

            return new_mask
    
        else :
            return mask
        
    def __is_a_contour(self, image : np.ndarray, 
                       background_proportion_thresh : np.float32 = 0.15):
        
        background_proportion = np.sum(image == 0)/(image.shape[0]*image.shape[1])
        return background_proportion > background_proportion_thresh
    
    def __thresholding(self,image : np.ndarray, threshold : int)-> np.ndarray:
        gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        ret, thresh_img = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY_INV)
        return thresh_img
    
    def __watershed_filter(self, image : np.ndarray, min_distance : int = 25)->np.ndarray:
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance,min_distance = min_distance, footprint=np.ones((3, 3)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        return labels
    
    def __eliminate_contour_epithelium(self, labels : np.ndarray,contour_pixels : np.ndarray, 
                                       intersection_threshold : np.float32 = 0.15):
        num_labels = np.max(labels)
        instance_pixels_epithelium_only = labels[contour_pixels]
        
        new_mask = labels.copy()
        for i in range(1,num_labels):
            epithelium_segmentation_out_of_tissue = np.sum(instance_pixels_epithelium_only == i)/np.sum(labels == i)
            if epithelium_segmentation_out_of_tissue > intersection_threshold :
                #define that instance as background
                new_mask[new_mask == i] = 0

        return new_mask
        
def main():
    PATH_TO_MASKS = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady/nuclei_filter_masks'
    PATH_TO_IMAGE = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady/patches'
    NEW_FOLDER = 'contour_masks'

    if NEW_FOLDER is None:
        #Replace mask into source folder
        DEST_FOLDER = PATH_TO_MASKS
    else :
        #Save fill hole mask in a new destination
        DEST_FOLDER = os.path.join(os.path.dirname(PATH_TO_MASKS),NEW_FOLDER)
        if not os.path.exists(DEST_FOLDER):
            os.mkdir(DEST_FOLDER)

    CHANNEL_EPITHELIUM = 1
    CHANNEL_NUCLEI = 2
    
    fn_path = os.listdir(PATH_TO_MASKS)
    contour_epithelium_filter = ContourEpitheliumRemoverFilter()

    for fn in fn_path:
        mask = cv.imread(os.path.join(PATH_TO_MASKS,fn))
        epithelium_mask = ((mask[:,:,CHANNEL_EPITHELIUM] > 120)*255).astype(np.uint8)
        image = cv.imread(os.path.join(PATH_TO_IMAGE,fn))

        #Resize image and mask
        resize_dim = 256
        image_reshaped = cv.resize(image.copy(), (resize_dim,resize_dim), interpolation = cv.INTER_CUBIC)
        mask_reshaped = ((cv.resize(epithelium_mask.copy(), (resize_dim,resize_dim), interpolation = cv.INTER_CUBIC) > 120)*255).astype(np.uint8)

        new_mask = contour_epithelium_filter.apply(image_reshaped,mask_reshaped)

        new_mask = cv.resize(new_mask.astype(np.uint8), (mask.shape[0],mask.shape[1]), interpolation = cv.INTER_CUBIC)  
        mask[:,:,CHANNEL_EPITHELIUM] = new_mask
        cv.imwrite(os.path.join(DEST_FOLDER,fn),mask)

if __name__ == "__main__":
    main()