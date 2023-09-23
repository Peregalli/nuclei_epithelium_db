import fast
import numpy as np
import os
from preprocessing.HoleFillFilterPO import HoleFillFilterPO

modelPath = "/home/agustina/fastpathology/datahub/colon-epithelium-segmentation-he-model/HE_IBDColEpi_512_2class_140222.onnx"
WSIPath = "/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/Doctorado operadas/44-pre.svs"
output_dir = 'colon_epithelium_tiffs'

WSI_fn = os.path.splitext(os.path.basename(WSIPath))[0]

#Hyperparameters
patchSize = 512
magnification = 10
overlapPercent = 0.3
scaleFactor=1.0
confidence_threshold = 0.2

importer = fast.WholeSlideImageImporter.create(WSIPath)

imageRenderer = fast.ImageRenderer.create().connect(importer)

tissueSegmentation = fast.TissueSegmentation.create(threshold = 70).connect(importer)

patchGenerator = fast.PatchGenerator.create(patchSize, 
                                            patchSize, 
                                            magnification=magnification, 
                                            overlapPercent=overlapPercent)\
                                                .connect(0, importer)\
                                                    .connect(1, tissueSegmentation)

# Create neural network object ###
nn = fast.NeuralNetwork.create(scaleFactor=scaleFactor,
                               modelFilename=modelPath).connect(0, patchGenerator)

HoleFilterProcess = HoleFillFilterPO.create(epithelium_threshold = 0.2,
                                            patchsize = patchSize).connect(0,nn)

# Create patch stitcher to generate pyramidal tiff output
stitcher = fast.PatchStitcher.create().connect(0, HoleFilterProcess)


# Create renderers to show both original WSI and segmentation output

wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green(), 2:fast.Color.Red()}).connect(stitcher)

# Create window to display segmentation
fast.SimpleWindow2D.create()\
        .connect(wsiRenderer)\
            .connect(segmentationRenderer).run()
    

finished = fast.RunUntilFinished.create()\
    .connect(stitcher)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output_dir,WSI_fn+'.tiff'))\
    .connect(finished)\
    .run()

