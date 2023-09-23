import fast
import time
import os

modelPath = "/home/agustina/fastpathology/datahub/nuclei-segmentation-model/high_res_nuclei_unet.onnx"
WSIPath = "/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/Doctorado operadas/44-D5.svs"
output_dir = 'nuclei_tiffs'
WSI_fn = os.path.splitext(os.path.basename(WSIPath))[0]

patchSize = 256

magnification = 20
overlapPercent = 0.1
scaleFactor= 1/255

importer = fast.WholeSlideImageImporter.create(WSIPath)

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

patchGenerator = fast.PatchGenerator.create(patchSize, 
                                            patchSize, 
                                            magnification=magnification, 
                                            overlapPercent=overlapPercent)\
    .connect(0, importer)\
    .connect(1, tissueSegmentation)

# Create neural network object
nn = fast.SegmentationNetwork.create(
    scaleFactor=scaleFactor,
    modelFilename=modelPath,
    # dimensionOrdering='channel-first'
    ).connect(0, patchGenerator)


# Create patch stitcher to generate pyramidal tiff output 
stitcher = fast.PatchStitcher.create().connect(0, nn)

# Create renderers to show both original WSI and segmentation output 
wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green()}).connect(stitcher)

# Create window to display segmentation###   
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
