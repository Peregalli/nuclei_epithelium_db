import fast

importer = fast.TIFFImagePyramidImporter.create('/home/agustina/Documents/FING/proyecto/WSI/IMÁGENES_BIOPSIAS/imágenes biopsias particulares/Lady.svs')
#
#/home/agustina/Documents/FING/proyecto/nuclei_epithelium_db/Lady_epithelium.tiff
segmentationRenderer = fast.SegmentationRenderer.create()\
    .connect(importer)

fast.SimpleWindow2D.create()\
    .connect(segmentationRenderer)\
    .run()

print('Hola')