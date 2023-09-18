import fast
import time
import os
from preprocessing.NucleiCleaningPO import NucleiCleaningPO

modelPath = "/home/agustina/fastpathology/datahub/nuclei-segmentation-model/high_res_nuclei_unet.onnx"
WSIPath = "/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs"
output_dir = 'nuclei_tiffs'
epitheliumPath = 'colon_epithelium_tiffs/Lady.tiff'
WSI_fn = os.path.splitext(os.path.basename(WSIPath))[0]

#Hyperparameters
patchSize = 256
magnification = 20
overlapPercent = 0.1
scaleFactor=1/255

importer = fast.WholeSlideImageImporter.create(WSIPath)

imageRenderer = fast.ImageRenderer.create().connect(importer)

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

segmentationImporter = fast.TIFFImagePyramidImporter.create(epitheliumPath)

patchGenerator = fast.PatchGenerator.create(patchSize, 
                                            patchSize, 
                                            magnification=magnification, 
                                            overlapPercent=overlapPercent)\
    .connect(0, importer)\
    .connect(1, tissueSegmentation)

#Create patcheGenerator for epithelium 
EpitheliumGenerator = fast.PatchGenerator.create(patchSize, 
                                            patchSize, 
                                            magnification=magnification, 
                                            overlapPercent=overlapPercent)\
                                                .connect(0, segmentationImporter)

# Create neural network object
nn = fast.NeuralNetwork.create(
    scaleFactor=scaleFactor,
    modelFilename=modelPath,
    # dimensionOrdering='channel-first'
    ).connect(0, patchGenerator)

#Import PO
NucleiFIlterProcess = NucleiCleaningPO.create().connect(0,nn).connect(1,EpitheliumGenerator)

# Create patch stitcher to generate pyramidal tiff output ###
stitcher = fast.PatchStitcher.create().connect(0, NucleiFIlterProcess)

# Create renderers to show both original WSI and segmentation output ###
wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green()}).connect(stitcher)

# Create window to display segmentation
fast.SimpleWindow2D.create()\
    .connect(wsiRenderer)\
    .connect(segmentationRenderer)\

finished = fast.RunUntilFinished.create()\
    .connect(stitcher)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output_dir,WSI_fn+'_.tiff'))\
    .connect(finished)\
    .run()
