import fast
import time
import os
import argparse

parser = argparse.ArgumentParser(description='Nuclei inference with high_res_nuclei_unet.onnx model.')
parser.add_argument('-m', '--model', help="model path. Called high_res_nuclei_unet.onnx, Available in datahub", type=str, default = 'models/high_res_nuclei_unet.onnx')
parser.add_argument('-w', '--wsi_path', help="path to wsi", type=str)
parser.add_argument('-o', '--output_folder', default = 'nuclei_tiffs', type=str)
parser.add_argument('-v', '--visualization', help = 'visualization render with FAST', action ="store_true", default = False)


def nuclei_segmentation_wsi(wsi_path : str , model_path : str = 'models/high_res_nuclei_unet.onnx', output : str = None, visualization : bool = False):

    WSI_fn = os.path.splitext(os.path.basename(wsi_path))[0]
    print(f'Inference started for {WSI_fn}, this could take a while...')

    #Hiperparameters
    patchSize = 256
    magnification = 20
    overlapPercent = 0.1
    scaleFactor= 1/255

    importer = fast.WholeSlideImageImporter.create(wsi_path)

    tissueSegmentation = fast.TissueSegmentation.create(threshold = 70).connect(importer)

    patchGenerator = fast.PatchGenerator.create(patchSize, patchSize, 
                                                magnification=magnification, 
                                                overlapPercent=overlapPercent)\
                                                    .connect(0, importer)\
                                                    .connect(1, tissueSegmentation)

    # Create neural network object
    nn = fast.SegmentationNetwork.create(
        scaleFactor=scaleFactor,
        modelFilename=model_path,
        # dimensionOrdering='channel-first'
        ).connect(0, patchGenerator)


    # Create patch stitcher to generate pyramidal tiff output 
    stitcher = fast.PatchStitcher.create().connect(0, nn)
    
    if visualization:
        # Create renderers to show both original WSI and segmentation output 
        wsiRenderer = fast.ImagePyramidRenderer.create().connect(0, importer)
        segmentationRenderer = fast.SegmentationRenderer.create(opacity = 0.1,borderOpacity=1, colors={1: fast.Color.Green()}).connect(stitcher)

        # Create window to display segmentation###   
        fast.SimpleWindow2D.create()\
            .connect(wsiRenderer)\
            .connect(segmentationRenderer).run()
    
    finished = fast.RunUntilFinished.create()\
        .connect(stitcher)
    
    if output is None:
        output = os.path.join(os.getcwd(),'nuclei_tiffs')

    if not os.path.exists(output):
        os.mkdir(output)
    
    exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output,WSI_fn+'.tiff'))\
        .connect(finished)\
        .run()
    print(f'Inference finished. Saved at {os.path.join(output,WSI_fn+".tiff")}')
    return

if __name__ == "__main__":

    args = parser.parse_args()

    nuclei_segmentation_wsi(args.wsi_path, args.model, args.output_folder, args.visualization)
