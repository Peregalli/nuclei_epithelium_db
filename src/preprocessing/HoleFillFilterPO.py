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
        self.createOutputPort(0)

        self.patchsize = patchsize
        self.threshold = epithelium_threshold
        self.prev_image = None
        self.epithelium_fill_filter = HoleFillFilter(kernel_size=int(0.02*self.patchsize))
        
    def execute(self):
        #Get Input 
        epithelium_prediction = self.getInputData(0)

        #Mask
        epithelium_mask = ((np.asarray(epithelium_prediction)[...,1]>self.threshold)*255).astype(np.uint8)

        #Postprocesing prediction
        epithelium_mask_fill = self.epithelium_fill_filter.apply(epithelium_mask)

        new_output_image = fast.Image.createFromArray((epithelium_mask_fill/255).astype(np.uint8))
        new_output_image.setSpacing(epithelium_prediction.getSpacing())
        self.addOutputData(0, new_output_image)