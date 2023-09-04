import fast
import time
import os

modelPath = "/home/agustina/fastpathology/datahub/nuclei-segmentation-model/high_res_nuclei_unet.onnx"
WSIPath = "/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs"
output_dir = 'nuclei_tiffs'
WSI_fn = os.path.splitext(os.path.basename(WSIPath))[0]

patchSize = 256

magnification = 20
overlapPercent = 0.1
scaleFactor=0.003921568627451

start = time.time()
importer = fast.WholeSlideImageImporter.create(WSIPath)

imageRenderer = fast.ImageRenderer.create().connect(importer)

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

patchGenerator = fast.PatchGenerator.create(patchSize, 
                                            patchSize, 
                                            magnification=magnification, 
                                            overlapPercent=overlapPercent)\
    .connect(0, importer)\
    .connect(1, tissueSegmentation)

print("Patch generator created")

### Batch patches ###
maxBatchSize = 1
batchGenerator = fast.ImageToBatchGenerator.create(maxBatchSize=maxBatchSize).connect(0, patchGenerator)

### Create neural network object ###
# TODO: Que es el scaleFactor? Por que vale 0.003921568627451?
nn = fast.SegmentationNetwork.create(
    scaleFactor=scaleFactor,
    modelFilename=modelPath,
    # dimensionOrdering='channel-first'
    ).connect(0, batchGenerator)

print("Segmentation Network created")

### Create patch stitcher to generate pyramidal tiff output ###
stitcher = fast.PatchStitcher.create().connect(0, nn)

print("Stitcher created")

### Create renderers to show both original WSI and segmentation output ###
wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green()}).connect(stitcher)

print("Renderers created")

### Create window to display segmentation###   
fast.SimpleWindow2D.create()\
    .connect(wsiRenderer)\
    .connect(segmentationRenderer)\
#    

end = time.time()
print(f"inference plus rendering took {end-start} seconds")

finished = fast.RunUntilFinished.create()\
    .connect(stitcher)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output_dir,WSI_fn+'.tiff'))\
    .connect(finished)\
    .run()
