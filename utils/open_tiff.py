import fast
import time


segmentationPath = "/home/agustina/Documents/FING/proyecto/QuickAnnotator/segmentation.tiff"


importer = fast.TIFFImagePyramidImporter.create(segmentationPath)

segmentationRenderer = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={1: fast.Color.Green()}).connect(importer)


fast.SimpleWindow2D.create().connect(segmentationRenderer).run()