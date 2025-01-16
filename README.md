![](https://github.com/Peregalli/MareFEst/blob/release/v1/data/logo_2.png)

MareFEst is an academic project within the context of the final engineering degree thesis from UdelaR (Universidad de la Republica - Uruguay).
Developed by N.Aguilera, A.Diaz and S.Zimmer

Original Title: [*Evaluación de la salud uterina en yeguas a partir de biopsias endometriales utilizando técnicas de procesamiento de imágenes y aprendizaje automático. Tesis de grado. Universidad de la República (Uruguay). Facultad de Ingeniería.*](https://www.colibri.udelar.edu.uy/jspui/handle/20.500.12008/43668)

## About the project

The purpose of this project is to investigate ways to incorporate image processing techniques and machine learning models in the study of endometrial biopsies from mares, stained with hematoxylin and eosin (H&E). The tissue samples available consist of Whole Slide Images (WSI): very high-resolution images that can easily reach 30,000 million pixels per channel and occupy several gigabytes of storage.

Animal fertility is a widely studied subject aimed at promoting the breeding of animals that inherit characteristics from their parents. Specially with equines, mares that possess valuable genetic material are used for equestrian sports. Unlike other production animals, equines are not selected for their reproductive characteristics, but rather for their athletic abilities or phenotypic traits.

Uterine health is crucial for a healthy gestation, so its important to know the state of a mare's uterus beforehand. Among other techniques, endometrial biopsies are performed for this purpose. In these biopsies, pathologists can study the presence and arrangement of different structures in the endometrium, estimating the potential fertility of the animal.

This project implements a sequential workflow covering the following tasks: performing inference with segmentation models on an entire WSI, aligning and slicing the WSI and the generated masks into patches, post-processing these patches, and extracting quantitative data. Among the applied machine learning models, two of them are pretrained with other domains (human tissues), and are use to segment glands and nuclei. The third model is a fibrosis segmentation model, which is one of the main pathologies affecting the ability of mares to carry a pregnancy to term. This model was trained with a database generated during the development of this project. Regarding the extracted quantitative data, exploration is done on gland density, and an attempt is made to classify nuclei into two classes: lymphocyte and non-lymphocyte.


## Pipeline
![](https://github.com/Peregalli/MareFEst/blob/release/v1/data/workflow.png)

## Get started!
1. Create a python virtual enviroment and install [fast](https://fast.eriksmistad.no/#get-started)
2. Install requirements as `pip install -r requirements.txt`
3. Download an example WSI from [here](https://drive.google.com/file/d/1kv4USkKeDXUmVU-yp3oR-uZbAtw5hX96/view?usp=sharing) and unzip it in `data` folder.

4. Explore running `Pipeline_Demo.ipynb`
