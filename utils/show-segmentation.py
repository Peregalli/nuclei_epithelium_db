import fast

WSIPath = "C:/Users/nicol/Documents/FING/proyecto/datos-majo/imagenes-biopsias/Doctorado operadas/44-D5.svs"
segmentationPath = "C:/Users/nicol/fastpathology/projects/nucleos/results/44-D5/Nuclei segmentation/Segmentation/Segmentation.tiff"

level = 0

assert(WSIPath != "")
assert(segmentationPath != "")

WSIImporter = fast.WholeSlideImageImporter\
    .create(WSIPath)

segmentationImporter = fast.TIFFImagePyramidImporter.create(segmentationPath)

### Create renderers to show both original WSI and segmentation output ###
WSIRenderer = fast.ImagePyramidRenderer.create().connect(0, WSIImporter)
segmentationRenderer = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={1: fast.Color.Green()}).connect(segmentationImporter)

### Create window to display segmentation###   
fast.SimpleWindow2D.create()\
    .connect(WSIRenderer)\
    .connect(segmentationRenderer)\
    .run()