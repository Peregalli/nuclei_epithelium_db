import fast
import numpy as np
import os
import sys
sys.path.append('src/')
from preprocessing.HoleFillFilterPO import HoleFillFilterPO
import argparse

parser = argparse.ArgumentParser(description='Epithelium inference with HE_IBDColEpi_512_2class_140222.onnx model.')
parser.add_argument('-m', '--model', help="model path. Called high_res_nuclei_unet.onnx, Available in datahub", type=str, default='models/HE_IBDColEpi_512_2class_140222.onnx')
parser.add_argument('-w', '--wsi_path', help="path to wsi", type=str)
parser.add_argument('-o', '--output_folder', default = 'colon_epithelium_tiffs', type=str)
parser.add_argument('-v', '--visualization', help = 'visualization render with FAST', action ="store_true", default = False)


def epithelium_segmentation_wsi(wsi_path : str , model_path : str = 'models/HE_IBDColEpi_512_2class_140222.onnx', output : str = None, visualization : bool = False, tissueSegmentationThreshold : int = 70):

    WSI_fn = os.path.splitext(os.path.basename(wsi_path))[0]

    #Hyperparameters
    patchSize = 512
    magnification = 10
    overlapPercent = 0.3
    scaleFactor=1.0

    # Configura el motor de inferencia para usar el orden de canales de PyTorch
    inferenceEngineList = fast.InferenceEngineManager.getEngineList()
    if (fast.InferenceEngineManager.isEngineAvailable(inferenceEngineList[0])):
        print(f"Engine {inferenceEngineList[0]} is available")
        inferenceEngine = fast.InferenceEngineManager.loadEngine(inferenceEngineList[0])
        inferenceEngine.setDeviceType(fast.InferenceDeviceType_GPU)
        inferenceEngine.setImageOrdering(fast.ImageOrdering_ChannelFirst)
    else:
        print(f"Engine {inferenceEngineList[0]} is not available")

    importer = fast.WholeSlideImageImporter.create(wsi_path)

    tissueSegmentation = fast.TissueSegmentation.create(threshold = tissueSegmentationThreshold).connect(importer)

    patchGenerator = fast.PatchGenerator.create(patchSize, 
                                                patchSize, 
                                                magnification=magnification, 
                                                overlapPercent=overlapPercent)\
                                                    .connect(0, importer)\
                                                        .connect(1, tissueSegmentation)

    # Create neural network object ###
    nn = fast.SegmentationNetwork.create(scaleFactor=scaleFactor,
                                   modelFilename=model_path).connect(0, patchGenerator)

    HoleFilterProcess = HoleFillFilterPO.create(epithelium_threshold = 0.2,
                                                patchsize = patchSize).connect(0,nn)

    # Create patch stitcher to generate pyramidal tiff output
    stitcher = fast.PatchStitcher.create().connect(0, HoleFilterProcess)


    # Create renderers to show both original WSI and segmentation output
    if visualization:
        wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
        segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green(), 2:fast.Color.Red()}).connect(stitcher)

        # Create window to display segmentation
        fast.SimpleWindow2D.create()\
                .connect(wsiRenderer)\
                    .connect(segmentationRenderer).run()


    finished = fast.RunUntilFinished.create()\
        .connect(stitcher)

    if output is None:
        output = os.path.join(os.getcwd(),'colon_epithelium_tiffs')
        
    if not os.path.exists(output):
        os.mkdir(output)

    exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output,WSI_fn+'.tiff'))\
        .connect(finished)\
        .run()
    print(f'Inference finished. Saved at {os.path.join(output,WSI_fn+".tiff")}')
    return
    
if __name__ == "__main__":

    args = parser.parse_args()

    epithelium_segmentation_wsi(args.wsi_path, args.model, args.output_folder, args.visualization)


