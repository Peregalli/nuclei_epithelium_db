import fast
import numpy as np
from preprocessing.NucleiCleaningFilter import NucleiCleaningFilter
import matplotlib.pyplot as plt


class NucleiCleaningPO(fast.PythonProcessObject):
    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createInputPort(1)
        self.createOutputPort(0)
        self.CHANNEL_EPITHELIUM = 1
        self.CHANNEL_NUCLEI = 2
        self.nuclei_cleaning_filter = NucleiCleaningFilter(nuclei_channel=self.CHANNEL_NUCLEI, epithelium_channel= self.CHANNEL_EPITHELIUM) 

    def execute(self):
        #Get prediction 
        prediction = self.getInputData(0)
        #epithelium = np.asarray(self.getInputData(1))
        #new_prediction = ((np.asarray(prediction)[...,1]>self.threshold)*255).astype(np.uint8)
        #mask = np.zeros((prediction .shape[0],prediction.shape[1],3))
        #mask[:,:,self.CHANNEL_EPITHELIUM] = epithelium
        #mask[:,:,self.CHANNEL_NUCLEI] = new_prediction
        #Postprocesing prediction
        #mask_fill = (self.nuclei_cleaning_filter.apply(mask)/255).astype(np.uint8)

        new_output_image = fast.Image.createFromArray(prediction)
        new_output_image.setSpacing(prediction.getSpacing())
        self.addOutputData(0, new_output_image)