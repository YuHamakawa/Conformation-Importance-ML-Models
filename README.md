# Conformation-Importance-ML-Models
Welcome to the official repository for the paper "Understanding Conformation Importance in Data-driven Property Prediction Models"!
This repository provide the data/code/model that used for the analysis.

## PQC Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13801221.svg)](https://doi.org/10.5281/zenodo.13801221)

One unique aspect of this study was creating the carefully controlled data sets for models’ performance evaluation in conformational diversity and the target property’s dependence on conformation.
For example, the QM9 dataset is limited to very small atoms with 9 or less heavy atoms, and many structurally abnormal molecules were observed. We hope that the PQC dataset will be used as a benchmark dataset for predicting properties using machine learning models.
Please download datasets from [here](https://zenodo.org/records/13801221)

## Installation
1. Install miniconda from [here](https://docs.anaconda.com/miniconda/)
2. Clone this repository:
```
git clone https://github.com/YuHamakawa/Conformation-Importance-ML-Models.git
```
3. Install required packeges:
```
cd Conformation-Importance-ML-Models
conda env create -f environment.yml
```
4. Download the PQC dataset and the APTCs dataset from [here](https://zenodo.org/records/13801221)
5. Unzip the downloaded file and place the files directly under the 'Conformation-Importance-ML-Models' directory.

## Tutorial
I provide code to reproduce the analysis in the paper as a notebook. The notebook consists of three files: one for creating data, one for building a model, and one for visualizing the results.
Check [tutorial_notebook](https://github.com/YuHamakawa/Conformation-Importance-ML-Models/tree/main/tutorial_notebook) directory for more detalis.


## Citation
Please kindly cite our paper if you use the data/code/model.

```
@dataset{hamakawa_2024_13801221,
  author       = {Hamakawa, Yu and
                  Miyao, Tomoyuki},
  title        = {{Datasets for understanding the importance of 
                   conformation in property prediction models}},
  month        = sep,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.13801221},
  url          = {https://doi.org/10.5281/zenodo.13801221}
}
```

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/YuHamakawa/Conformation-Importance-ML-Models/blob/main/LICENSE) for additional details.