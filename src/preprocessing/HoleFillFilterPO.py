import fast
import numpy as np
from preprocessing.HoleFillFilter import HoleFillFilter
from preprocessing.NucleiCleaningFilter import NucleiCleaningFilter
from preprocessing.ContourEpitheliumRemoverFilter import ContourEpitheliumRemoverFilter
import matplotlib.pyplot as plt


class HoleFillFilterPO(fast.PythonProcessObject):
    def __init__(self, epithelium_threshold,patchsize,channel_nuclei = 2,channel_epithelium = 1):
        super().__init__()
        self.createInputPort(0)
        self.createInputPort(1)
        self.createOutputPort(0)
        self.createInputPort(2)
        self.patchsize = patchsize
        self.threshold = epithelium_threshold
        self.channel_nuclei =  channel_nuclei
        self.channel_epithelium = channel_epithelium
        self.prev_image = None
        self.epithelium_fill_filter = HoleFillFilter(kernel_size=int(0.02*self.patchsize))
        self.nuclei_cleaning_filter = NucleiCleaningFilter(nuclei_channel=channel_nuclei, epithelium_channel= channel_epithelium)     
        self.contour_epithelium_filter = ContourEpitheliumRemoverFilter(kernel_size_erode = int(0.02*self.patchsize), watershed_min_distance= int(0.1*patchsize))

    def execute(self):
        #Get Input 
        epithelium_prediction = self.getInputData(0)
        image = np.asarray(self.getInputData(1))
        nuclei_mask = np.asarray(self.getInputData(2))[:,:,0]

        #Mask
        epithelium_mask = ((np.asarray(epithelium_prediction)[...,1]>self.threshold)*255).astype(np.uint8)

        #Postprocesing prediction
        epithelium_mask_fill = self.epithelium_fill_filter.apply(epithelium_mask)

        #Prepare nuclei mask
        #mask = np.zeros((self.patchsize,self.patchsize,3))
        #mask[:,:,self.channel_nuclei] = nuclei_mask
        #mask[:,:,self.channel_epithelium] = epithelium_mask_fill
        #mask[:,:,self.channel_nuclei] = self.nuclei_cleaning_filter.apply(mask)[:,:,self.channel_nuclei]
        #mask[:,:,self.channel_epithelium] = self.contour_epithelium_filter.apply(image,mask[:,:,self.channel_epithelium])
        #mask = np.argmax(mask)
        new_output_image = fast.Image.createFromArray((epithelium_mask).astype(np.uint8))
        new_output_image.setSpacing(epithelium_prediction.getSpacing())
        self.addOutputData(0, new_output_image)