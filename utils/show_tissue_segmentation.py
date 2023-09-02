import fast
import os

WSIPath = "/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs"
output_dir = 'tissue_segmentation'
WSI_fn = os.path.splitext(os.path.basename(WSIPath))[0]

importer = fast.WholeSlideImageImporter\
    .create(WSIPath)

tissueSegmentation = fast.TissueSegmentation.create()\
    .connect(importer)

renderer = fast.ImagePyramidRenderer.create()\
    .connect(importer)

segmentationRenderer = fast.SegmentationRenderer.create()\
    .connect(tissueSegmentation)

fast.SimpleWindow2D.create()\
    .connect(renderer)\
    .connect(segmentationRenderer).run()