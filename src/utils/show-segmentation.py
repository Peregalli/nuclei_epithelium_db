import fast

WSIPath = '/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs'
segmentationPath = 'colon_epithelium_tiffs/Lady.tiff'
segmentationPath_2 = 'nuclei_tiffs/Lady.tiff'

level = 0

assert(WSIPath != "")
assert(segmentationPath != "")

WSIImporter = fast.WholeSlideImageImporter\
    .create(WSIPath)

segmentationImporter = fast.TIFFImagePyramidImporter.create(segmentationPath)
segmentationImporter_2 = fast.TIFFImagePyramidImporter.create(segmentationPath_2)

### Create renderers to show both original WSI and segmentation output ###
WSIRenderer = fast.ImagePyramidRenderer.create().connect(0, WSIImporter)
segmentationRenderer_1 = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={0: fast.Color.Green()}).connect(0,segmentationImporter)
segmentationRenderer_2 = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={0: fast.Color.Red()}).connect(segmentationImporter_2)

### Create window to display segmentation###   
fast.SimpleWindow2D.create()\
    .connect(WSIRenderer)\
    .connect(segmentationRenderer_1)\
    .run()