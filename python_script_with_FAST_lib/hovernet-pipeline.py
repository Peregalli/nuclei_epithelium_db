import fast
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

WSIPath = "/home/sofia/Documents/FING/Proyecto/Muestras/Buenas/Lady.svs"
patchSize = 256

# https://github.com/vqdang/hover_net#usage-and-options-1
#'fast' model mode uses a 256x256 patch input and 164x164 patch output.

level = 0

# The PanNuke Dataset: "We re-sized the selected visual fields so that all images were at 40× resolution"
# In this link info about level magnification can be found: https://fast.eriksmistad.no/python-tutorial-wsi.html 
# Level 0 in the pyramid is the full resolution image W x H, while the next level 1 is the same image but with a reduced size, 
# typically half the width and height of the previous level (W/2 x H/2).

importer = fast.WholeSlideImageImporter\
    .create(WSIPath)

imageRenderer = fast.ImageRenderer.create().connect(importer)

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

patchGenerator = fast.PatchGenerator.create(patchSize, patchSize, level=level, overlapPercent=0.1)\
    .connect(0, importer)\
    .connect(1, tissueSegmentation)

print("Patch generator created")

### Show generated patches ###
# Create a 3x3 subplot for every set of 9 patches
# patch_list = []
# for patch in fast.DataStream(patchGenerator):
#     patch_list.append(patch)
#     if len(patch_list) == 9:
#         # Display the 9 last patches
#         f, axes = plt.subplots(3,3, figsize=(10,10))
#         for i in range(3):
#             for j in range(3):
#                 axes[i, j].imshow(patch_list[i + j*3])
#         plt.show()
#         patch_list.clear()

### Save generated patches to disk ###
# patch_list = []
# for i, patch in enumerate(fast.DataStream(patchGenerator)):
#     filename = 'C:/Users/nicol/Documents/FING/proyecto/fastpathology/lady-p' + str(patchSize) + '-m' + str(magnification) + str(f'-{i}.png')
#     imageExporter = fast.ImageFileExporter.create(filename)
#     imageExporter.setInputData(patch)
#     imageExporter.update()

### Batch patches ###
maxBatchSize = 1
batchGenerator = fast.ImageToBatchGenerator.create(maxBatchSize=maxBatchSize).connect(0, patchGenerator)

### Create neural network object ###
modelPath = "/home/sofia/Documents/FING/Proyecto/hover_net/hovernet_pannuke.onnx"
nn = fast.NeuralNetwork.create(
    scaleFactor=0.003921568627451,
    modelFilename=modelPath,
    # dimensionOrdering='channel-first'
    ).connect(0, batchGenerator)

print("Neural network created")

### Convert output to segmentation ###
converter = fast.TensorToSegmentation.create().connect(0, nn)

print("Converter created")

### Create patch stitcher to generate pyramidal tiff output ###
stitcher = fast.PatchStitcher.create(patchesAreCropped=True).connect(0, converter)

print("Stitcher created")

### Create renderers to show both original WSI and segmentation output ###
wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
segmentationRenderer = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={1: fast.Color.Black(), 2: fast.Color.Red(), 3: fast.Color.Green(), 4: fast.Color.Blue(), 5: fast.Color.Yellow(), 6: fast.Color.White()},).connect(stitcher)

### Create labels###
labelRenderer = fast.SegmentationLabelRenderer.create(labelNames={1: 'nolabe', 2: 'neopla', 3: 'inflam', 4: 'connec', 5: 'necros', 6: 'no-neo'}, labelColors={1: fast.Color.Black(), 2: fast.Color.Red(), 3: fast.Color.Green(), 4: fast.Color.Blue(), 5: fast.Color.Yellow(), 6: fast.Color.White()},).connect(nn)


### Create window to display segmentation###   
fast.SimpleWindow2D.create()\
    .connect(wsiRenderer)\
    .connect([imageRenderer, segmentationRenderer, labelRenderer])\
    .run()