import fast
import time

modelPath = "C:/Users/nicol/fastpathology/models/high_res_nuclei_unet.onnx"
WSIPath = "C:/Users/nicol/Documents/FING/proyecto/datos-majo/imagenes-biopsias/Doctorado operadas/44-D5.svs"
segmentationPath = "C:/Users/nicol/fastpathology/projects/nucleos/results/44-D5/Nuclei segmentation/Segmentation/Segmentation.tiff"

patchSize = 256

magnification = 20

start = time.time()
importer = fast.WholeSlideImageImporter\
.create(WSIPath)

imageRenderer = fast.ImageRenderer.create().connect(importer)

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

patchGenerator = fast.PatchGenerator.create(patchSize, patchSize, magnification=magnification, overlapPercent=0.1)\
    .connect(0, importer)\
    .connect(1, tissueSegmentation)

print("Patch generator created")

### Batch patches ###
maxBatchSize = 1
batchGenerator = fast.ImageToBatchGenerator.create(maxBatchSize=maxBatchSize).connect(0, patchGenerator)

### Create neural network object ###
# TODO: Que es el scaleFactor? Por que vale 0.003921568627451?
nn = fast.SegmentationNetwork.create(
    scaleFactor=0.003921568627451,
    modelFilename=modelPath,
    # dimensionOrdering='channel-first'
    ).connect(0, batchGenerator)

print("Segmentation Network created")

### Create patch stitcher to generate pyramidal tiff output ###
stitcher = fast.PatchStitcher.create().connect(0, nn)

print("Stitcher created")

### Create renderers to show both original WSI and segmentation output ###
wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
segmentationRenderer = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={1: fast.Color.Green()}).connect(stitcher)

print("Renderers created")

### Create window to display segmentation###   
fast.SimpleWindow2D.create()\
    .connect(wsiRenderer)\
    .connect(segmentationRenderer)\
    .run()

end = time.time()
print(f"inference plus rendering took {end-start} seconds")