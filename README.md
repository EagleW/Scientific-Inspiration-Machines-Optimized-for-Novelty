# Learning to Generate Novel Scientific Directions with Contextualized Literature-based Discovery


Table of Contents
=================

* [Overview](#overview)
  
* [Requirements](#requirements)
  
* [Quickstart](#quickstart)
  
* [Citation](#citation)

## Overview

<p align="center">
  <img src="https://eaglew.github.io/images/ai2.png?raw=true" alt="Photo" style="width: 100%;"/>
</p>

## Requirements

### Environment

* Python 3.8.5
  
* Ubuntu 22.04


### Environment Setup Instructions

To set up the environment for this repository, please follow the steps below:

Step 1: Create a Python environment (optional)
If you wish to use a specific Python environment, you can create one using the following:

```bash
conda create -n pyt1.11 python=3.8.5
```

Step 2: Install PyTorch with CUDA (optional)
If you want to use PyTorch with CUDA support, you can install it using the following:

```bash
conda install pytorch==1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Step 3: Install Python dependencies
To install the required Python dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Data Setup

1. Unzip all the zip files located in the data folder, including its subfolders.
2. Place the following folders, extracted from their respective zip files, under the data folder: `kg`,`ct`, and `gold_subset`
3. Locate the `local_context_dataset` folder unzipped from `data/idea-sentence/local_context_dataset.zip`.Move it to `idea-sentence/models/T5`.
4. Find the `local_dataset` folder unzipped from `data/idea-node/local_dataset.zip`. Place them in `idea-node/models/Dual_Encoder`.
5. Copy the file `e2t.json` and paste it into the following folders:  `idea-node/models/GPT3.5*/`, `idea-node/preprocess/`, `idea-sentence/models/GPT3.5*/`, and `idea-sentence/preprocess/`
  
### Data Preprocess

1. Navigate to the `idea-node/preprocess` and run the `bash preprocess.sh`
2. Navigate to the `idea-sentence/preprocess` and run the `bash preprocess.sh` 

### Data and Code Description

The project data includes the following components:

1. `data/local_context_dataset`: This folder contains the training, validation, and testing files for idea sentence generation.
2. `data/local_dataset`: This folder contains the training, validation, and testing files for idea node prediction.
3. `data/kg/*.json`: The `data/kg` directory contains files that store the original Information Extraction (IE) results for all paper abstracts.
4. `data/ct/*.csv`: The `data/ct` directory contains files that represent the citation network for all papers.
5. `data/gold_subset`: This directory contains our gold annotation subsets.
6. `idea-node/evaluation` and `idea-sentence/evaluation` contain sample evaluation code.


### Results

`result/sentence_generation.zip`: This zip file contains all GPT3.5/GPT4 results for idea-sentence generation task

## Quickstart

### Training

To train the model under `*\models\*`, run the following command:

```bash
bash finetune_*.sh 
```

#### Test

To test the model under `*\models\*`, run the following command:


```bash
bash eval_*.sh 
```

## Citation

```
@article{wang2023learning,
  title={Learning to Generate Novel Scientific Directions with Contextualized Literature-based Discovery},
  author={Wang, Qingyun and Downey, Doug and Ji, Heng and Hope, Tom},
  journal={arXiv preprint arXiv:2305.14259},
  year={2023}
}
```