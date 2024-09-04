# Conformation-Importance-ML-Models
Welcome to the official repository for the paper "Understanding Conformation Importance in Data-driven Property Prediction Models"!
This repository provide the data/code/model that used for the analysis.

## PQC Dataset
One unique aspect of this study was creating the carefully controlled data sets for models’ performance evaluation in conformational diversity and the target property’s dependence on conformation.
For example, the QM9 dataset is limited to very small atoms with 9 or less heavy atoms, and many structurally abnormal molecules were observed. We hope that the PQC dataset will be used as a benchmark dataset for predicting properties using machine learning models.
Check "dataset" directory for more detalis.

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

## Tutorial
I provide code to reproduce the analysis in the paper as a notebook. The notebook consists of three files: one for creating data, one for building a model, and one for visualizing the results.
Check "tutorial_notebook" directory for more detalis.

## Result
The "output" directory provides the model results that are presented in the tables of the paper.

## Citation
Please kindly cite our paper if you use the data/code/model.

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/YuHamakawa/Conformation-Importance-ML-Models/blob/main/LICENSE) for additional details.