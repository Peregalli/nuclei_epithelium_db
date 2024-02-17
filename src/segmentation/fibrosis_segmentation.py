import fast
import time
import argparse
import os

import matplotlib.pyplot as plt

patchSize = 1000
resize = 0.5
level = 1

def fibrosis_segmentation_wsi(wsi_path : str , model_path : str = 'models/fibrosis.onnx', output_dir : str = None, visualization : bool = False, tissueSegmentationThreshold : int = 85):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(),'fibrosis_tiffs')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Nombre del archivo de salida
    segmentationFileName = os.path.splitext(os.path.basename(wsi_path))[0]+'.tiff'

    # Configura el motor de inferencia para usar el orden de canales de PyTorch
    inferenceEngineList = fast.InferenceEngineManager.getEngineList()
    if (fast.InferenceEngineManager.isEngineAvailable(inferenceEngineList[0])):
        print(f"Engine {inferenceEngineList[0]} is available")
        inferenceEngine = fast.InferenceEngineManager.loadEngine(inferenceEngineList[0])
        inferenceEngine.setDeviceType(fast.InferenceDeviceType_GPU)
        inferenceEngine.setImageOrdering(fast.ImageOrdering_ChannelFirst)
    else:
        print(f"Engine {inferenceEngineList[0]} is not available")

    # Importa una WSI
    importer = fast.WholeSlideImageImporter.create(wsi_path)

    # Segmenta el tejido del fondo
    wsiRenderer = fast.ImagePyramidRenderer.create().connect(importer)
    tissueSegmentation = fast.TissueSegmentation.create(threshold=tissueSegmentationThreshold).connect(importer)

    # Parte la WSI en parches
    patchGenerator = fast.PatchGenerator.create(patchSize, patchSize, level=1, overlapPercent=0.1).connect(0,importer).connect(1, tissueSegmentation)
    # patchGenerator = fast.PatchGenerator.create(args.patchSize, args.patchSize, level=1, overlapPercent=0.1).connect(0,importer)

    # Redimensiona los parches
    imageResizer = fast.ImageResizer.create(int(patchSize*resize), int(patchSize*resize), useInterpolation=True).connect(0, patchGenerator)

    # Genera un batch de parches
    maxBatchSize = 1
    batchGenerator = fast.ImageToBatchGenerator.create(maxBatchSize=maxBatchSize).connect(0, imageResizer)

    # Crea una red neuronal a partir de un archivo ONNX
    nn = fast.NeuralNetwork.create(scaleFactor=0.003921568627451,modelFilename=model_path).connect(0, batchGenerator)

    # Convierte el tensor de inferencia a una segmentacion
    tensorToSegmentation = fast.TensorToSegmentation.create(hasBackgroundClass=True).connect(0, nn)
    
    # Redimensiona los parches inferidos a la resolucion original
    imageResizer = fast.ImageResizer.create(int(patchSize), int(patchSize), useInterpolation=True).connect(0, tensorToSegmentation)
    
    # Junta la inferencia de los parches en un solo archivo TIFF
    stitcher = fast.PatchStitcher.create(patchesAreCropped=False).connect(0, imageResizer)

    finished = fast.RunUntilFinished.create().connect(stitcher)
    
    # Guarda el TIFF a disco
    exporter = fast.TIFFImagePyramidExporter.create(os.path.join(output_dir, segmentationFileName)).connect(finished)

    if visualization:
        ### Create window and renderers to display segmentation###
        wsiRenderer = fast.ImagePyramidRenderer.create().connect(importer)
        segmentationRenderer = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={1: fast.Color.Green()}).connect(stitcher)
        tissueSegmentationRenderer = fast.SegmentationRenderer.create(borderOpacity=0.5, colors={1: fast.Color.Red()}).connect(tissueSegmentation)
        # fast.SimpleWindow2D.create().connect(wsiRenderer).connect(segmentationRenderer).run()
        fast.SimpleWindow2D.create().connect(wsiRenderer).connect(segmentationRenderer).connect(tissueSegmentationRenderer).run()
    else:
        exporter.run()
        print(f'Inference finished. Saved at {os.path.join(output_dir, segmentationFileName)}')

if __name__ == "__main__":

    # ----- parse command line arguments
    parser = argparse.ArgumentParser(description='Script to convert QuickAnnotator\'s UNet from PyTorch format to ONNX format')

    parser.add_argument('-m', '--modelPath', help="Path to ONNX model", default='', type=str)
    parser.add_argument('-w', '--wsiPath', help="Path to input WSI", default='', type=str)
    parser.add_argument('-p', '--patchSize', help="Size of input in pixels", default=0, type=int)

    args = parser.parse_args()
    print(f"args: {args}")

    try:
        assert(args.modelPath != '')
        assert(args.wsiPath != '')
        assert(args.patchSize > 0)

        start = time.time()

        fibrosis_segmentation_wsi(args.wsiPath, args.modelPath, visualization=True)

        end = time.time()
        print(f"inference plus rendering took {end-start} seconds")
    except Exception as e:
        print(e)
        print("Error")