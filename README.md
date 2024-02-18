# SciMON: Scientific Inspiration Machines Optimized for Novelty


Table of Contents
=================

* [Overview](#overview)
  
* [Requirements](#requirements)
  
* [Quickstart](#quickstart)
  
* [Citation](#citation)

## Overview

<p align="center">
  <img src="https://eaglew.github.io/images/ai2_.png?raw=true" alt="Photo" style="width: 100%;"/>
</p>

## Requirements

### Environment 

* Python 3.8.5
  
* Ubuntu 22.04

### Environment Setup Instructions for NLP

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

### Environment Setup Instructions for biochemical

To set up the environment for this repository, please follow the steps below:

Step 1: Create a Python environment (optional)
If you wish to use a specific Python environment, you can create one using the following:

```bash
conda create -n pyt2.2 python=3.10
```

Step 2: Install PyTorch with CUDA (optional)
If you want to use PyTorch with CUDA support, you can install it using the following:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Step 3: Install Python dependencies
To install the required Python dependencies, run the following command:

```bash
pip install -r requirements1.txt
```

### Data Setup

1. Unzip all the zip files located in the data folder, including its subfolders.
2. Place the following folders, extracted from their respective zip files, under the data folder: `kg`,`ct`, and `gold_subset`
3. Locate the `local_context_dataset` folder unzipped from `data/local_context_dataset.zip`.Move it to `models/T5`.
4. Copy the file `e2t.json` and paste it into the following folders:   `models\GPT*\`, `models\Iterative\`, and `preprocess\`
5. Locate the `og`, `sim`, `kg`, and `ct` folder under the biomedical folder, copy them to the corresponding folder under `biomedical_models\*\data`
  
### Data Preprocess

1. Navigate to the `preprocess` and run the `bash preprocess.sh`
2. Navigate to the `models\GPTRND` and run `preprocess.py`
3. Navigate to the `biomedical_models\*` and run `preprocess.py`

### Data and Code Description

The project data includes the following components:

1. `data/local_context_dataset.zip`: This folder contains the training, validation, and testing files for our task.
2. `data/kg/*.json`: The `data/kg` directory contains files that store the original Information Extraction (IE) results for all paper abstracts.
3. `data/ct/*.csv`: The `data/ct` directory contains files that represent the citation network for all papers.
4. `data/gold_subset`: This directory contains our gold annotation subsets.
5. `data/biomedical.zip`: This directory contains our biochemical datasets.
6. `evaluation` contain sample evaluation code.

### Results

`result/sentence_generation.zip`: This zip file contains part of GPT3.5/GPT4 results

## Quickstart for NLP domain

Set up environment first:

```bash
conda activate pyt1.11
```

### Training

To train the T5 model under `models\T5*`, run the following command:

```bash
bash finetune_*.sh 
```

#### Test

To test the T5 model under `models\T5*`, run the following command:

```bash
bash eval_*.sh 
```

To test the GPT3.5 model under `models\GPT*`, run the following command:

```bash
bash eval3.sh 
```

After getting GPT3.5 results, we can also get GPT4 results using same input by running the following command:

```bash
python gpt4.py
```

After gettubg GPT4 results, first copy all GPT4 results under the `iterative` folder, you can then run the first two iterations of iterative novelty boosting by running the following command: 

```bash
python calculate_sim.py
python gpt4_iter1.py
python calculate_sim1.py
python gpt4_iter2.py
```

## Quickstart for biochemical domain

Set up environment first:

```bash
conda activate pyt2.2
```

Download `Meditron-7b` model from huggingface and put it under `biomedical_models\model`

### Training

To train the T5 model under `biomedical_models\*\`, run the following command:

```bash
bash train.sh 
```

#### Test
To test the trained model under `biomedical_models\*\`, run the following command:

```bash
python inf_generator.py 
```

# Citation
```
@article{wang2023learning,
  title={SciMON: Scientific Inspiration Machines Optimized for Novelty},
  author={Wang, Qingyun and Downey, Doug and Ji, Heng and Hope, Tom},
  journal={arXiv preprint arXiv:2305.14259},
  year={2023}
}
```