import fast
import numpy as np
import os
from preprocessing.HoleFillFilterPO import HoleFillFilterPO
import argparse

parser = argparse.ArgumentParser(description='Inference with glans-segmentation.onnx model.')
parser.add_argument('-m', '--model', help="model path. Called high_res_nuclei_unet.onnx, Available in datahub", type=str)
parser.add_argument('-w', '--wsi_path', help="path to wsi", type=str)
parser.add_argument('-o', '--output_folder', default = 'gland_tiffs', type=str)


def gland_segmentation_wsi(wsi_path : str , model_path : str, output : str):

    WSI_fn = os.path.splitext(os.path.basename(wsi_path))[0]

    #Hyperparameters
    patchSize = 512
    magnification = 20
    overlapPercent = .0
    scaleFactor=1/255

    importer = fast.WholeSlideImageImporter.create(wsi_path)

    tissueSegmentation = fast.TissueSegmentation.create(threshold = 70).connect(importer)

    patchGenerator = fast.PatchGenerator.create(patchSize, 
                                                patchSize, 
                                                magnification=magnification, 
                                                overlapPercent=overlapPercent)\
                                                    .connect(0, importer)\
                                                        .connect(1, tissueSegmentation)
    
    #maxBatchSize = 2
    #batchGenerator = fast.ImageToBatchGenerator.create(maxBatchSize=maxBatchSize).connect(0, patchGenerator)

    nn = fast.SegmentationNetwork.create(scaleFactor=scaleFactor,
                                   modelFilename=model_path).connect(0, patchGenerator)

    HoleFilterProcess = HoleFillFilterPO.create(epithelium_threshold = 0.2,
                                                patchsize = patchSize).connect(0,nn)

    # Create patch stitcher to generate pyramidal tiff output
    stitcher = fast.PatchStitcher.create().connect(0, HoleFilterProcess)


    # Create renderers to show both original WSI and segmentation output

    wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
    segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green(), 2:fast.Color.Red()}).connect(stitcher)

    # Create window to display segmentation
    fast.SimpleWindow2D.create()\
            .connect(wsiRenderer)\
                .connect(segmentationRenderer).run()


    finished = fast.RunUntilFinished.create()\
        .connect(stitcher)

    if not os.path.exists(output):
        os.mkdir(output)

    exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output,WSI_fn+'.tiff'))\
        .connect(finished)\
        .run()
    
    return
    
if __name__ == "__main__":

    args = parser.parse_args()

    gland_segmentation_wsi(args.wsi_path, args.model, args.output_folder)


