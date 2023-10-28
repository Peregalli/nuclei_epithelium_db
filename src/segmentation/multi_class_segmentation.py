import os
import sys
sys.path.append('src/')
from tqdm import tqdm  
import argparse
import time
import csv


parser = argparse.ArgumentParser(description='Run segmentation models on a set of files')
parser.add_argument('-t', '--files_to_segment',type = str)
parser.add_argument('-c', '--colon_epithelium', action ="store_true", default = False)
parser.add_argument('-g', '--glands_epithelium', action ="store_true", default = False)
parser.add_argument('-n', '--nuclei_epithelium', action ="store_true", default = False)


def segment_models(file_names, colon_epithelium : bool = None, glands_epithelium : bool = None, nuclei_epithelium : bool = None):


    for file_path in tqdm(file_names):

        file_path = file_path[0]
        if not os.path.exists(file_path):
            raise ValueError('File does not exist')
    
        else : 
            start_time = time.time()

            if colon_epithelium :
                print('Running colon epithelium segmentation')
                os.system(f'python src/segmentation/colon_epithelium_segmentation_with_postprocessing.py -w {file_path}')
                print('Done epitheliun segmentation')

            if glands_epithelium :
                print('Running gland segmentation')
                os.system(f'python src/segmentation/gland_segmentation_with_postprocessing.py -w {file_path}')
                print('Done gland segmentation')

            if nuclei_epithelium :
                print('Running nuclei segmentation')
                os.system(f'python src/segmentation/nuclei_segmentation.py -w {file_path}')
                print('Done nuclei segmentation')
            
            fname = os.path.basename(file_path)
            end_time = time.time()
            elapsed_time = (end_time - start_time)/60
            print(f'Elapsed time : {elapsed_time:.2f} min for {fname}')
            print('Done {file_path} segmentation')


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.files_to_segment, 'r') as file:
        csv_reader = csv.reader(file)

        # Convert the CSV data into a list
        data_list = list(csv_reader)

    segment_models(data_list, args.colon_epithelium, args.glands_epithelium, args.nuclei_epithelium)


        