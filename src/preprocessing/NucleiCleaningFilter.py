import numpy as np
import cv2 as cv
import os

class NucleiCleaningFilter():
    def __init__(self,nuclei_channel,epithelium_channel):
        self.nuclei_channel = nuclei_channel 
        self.epithelium_channel = epithelium_channel
        self.kernel_size = 3
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.glands_proportion = 0.0035
        self.num_nuclei = None
    
    def apply(self, mask : np.ndarray)-> np.ndarray:

        nuclei_mask = ((mask[:,:,self.nuclei_channel]> 120)*255).astype(np.uint8)
        epithelium_mask = ((mask[:,:,self.epithelium_channel]> 120)*255).astype(np.uint8)
        w, h, _ = mask.shape

        new_nuclei_mask = nuclei_mask.copy()
        new_nuclei_mask[epithelium_mask == 255] = 0
        nuclei_mask_dilate = cv.dilate(nuclei_mask, self.kernel, iterations=2)
        num_labels, nuceli_instances, stats, centroids = cv.connectedComponentsWithStats(nuclei_mask_dilate)

        # if dilate nuclei are is bigger that glands_proportion threshold then those are nuclei glands
        nuclei_glands = np.where(stats[:,4] >= self.glands_proportion*(w*h))[0]

        for nuclei in nuclei_glands:
            new_nuclei_mask[nuceli_instances == nuclei] = 0
        
        mask[:,:,self.nuclei_channel] = new_nuclei_mask

        # Calculate the total number of nuclei
        self.num_nuclei = num_labels - len(nuclei_glands) - 1

        return mask

def main():
    PATH_TO_MASKS = '/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady/fill_masks'
    NEW_FOLDER = 'nuclei_filter_masks'

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
    
    fn_path = os.listdir(PATH_TO_MASKS)
    nuclei_cleaning_filter = NucleiCleaningFilter(nuclei_channel=CHANNEL_NUCLEI, epithelium_channel= channel_glands)

    for fn in fn_path:
        mask = cv.imread(os.path.join(PATH_TO_MASKS,fn))  
        new_mask = nuclei_cleaning_filter.apply(mask)
        cv.imwrite(os.path.join(DEST_FOLDER,fn),new_mask)

if __name__ == "__main__":
    main()