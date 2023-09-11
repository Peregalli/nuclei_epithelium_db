import fast
import numpy as np
from preprocessing.HoleFillFilter import HoleFillFilter


class PosprocessingPO(fast.PythonProcessObject):
    def __init__(self, threshold):
        super().__init__()
        self.createInputPort(0)
        self.createInputPort(1)
        self.createOutputPort(0)
        self.threshold = threshold
        self.epithelium_fill_filter = HoleFillFilter()   

    def execute(self):
        #Get prediction 
        prediction = self.getInputData(0)
        image = np.asarray(self.getInputData(1))
        new_prediction = ((np.asarray(prediction)[...,1]>self.threshold)*255).astype(np.uint8)

        #Postprocesing prediction
        mask_fill = self.epithelium_fill_filter.apply(new_prediction)
        
        new_output_image = fast.Image.createFromArray(mask_fill)
        new_output_image.setSpacing(prediction.getSpacing())
        self.addOutputData(0, new_output_image)