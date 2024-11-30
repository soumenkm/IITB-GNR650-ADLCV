
# Zero-Shot Learning (ZSL) Model

This repository contains the implementation of a zero-shot learning (ZSL) model, evaluating different configurations of visual feature extractors (ResNet-50 and ViT) and text embedding models (Word2Vec and FastText) for unseen class prediction. This README provides instructions for setting up, running the model, and reproducing the results.

## Table of Contents
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Code Structure](#code-structure)
- [Running the Model](#running-the-model)
- [Reproducing Results](#reproducing-results)
- [Results](#results)

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure that you have Python 3.12+ installed.

## Dataset Preparation

Download the AwA2 dataset used for zero-shot learning. 
```
wget "https://cvml.ista.ac.at/AwA2/AwA2-data.zip
unzip AwA2-data.zip -d ./Assignment2/data
```

## Code Structure

- `dataset.py`: Handles loading and preprocessing the dataset, as well as splitting classes into seen and unseen categories.
- `main.py`: The primary script to run training and evaluation. It coordinates data loading, model initialization, training, and evaluation.
- `text-encoder.py`: Defines the text embedding model used to generate class vectors, either using Word2Vec or FastText.
- `visual-encoder.py`: Defines the visual encoder (ResNet-50 or ViT) for extracting feature representations from images.

## Running the Model

To train and evaluate the model, use `main.py` with specific configuration flags. Below are example commands for each configuration:

1. **ResNet-50 + Word2Vec**
   ```bash
   python Assignment2/main.py --visual_encoder resnet50 --text_encoder word2vec
   ```

2. **ResNet-50 + FastText**
   ```bash
   python Assignment2/main.py --visual_encoder resnet50 --text_encoder fasttext
   ```

3. **ViT Base + Word2Vec**
   ```bash
   python Assignment2/main.py --visual_encoder vit --text_encoder word2vec
   ```

4. **ViT Base + FastText**
   ```bash
   python Assignment2/main.py --visual_encoder vit --text_encoder fasttext
   ```

## Reproducing Results

The precomputed visual features are large so these features can be downloaded from https://drive.google.com/drive/folders/1KLftsJcg09cVrwJ9I8sdTfYq42RAjYUu?usp=share_link. Once downloaded, save the pkl files in the directory `Assignment2/data/visual_features`. If you wish to recompute the visual and text features then please run the file `visual_encoder.py` and `text_encoder.py` by changing the configurations.

To reproduce the results as reported:
1. Ensure you are using the same dataset and structure as specified in [Dataset Preparation](#dataset-preparation).
2. Run each of the above configurations and note the accuracy results on unseen classes.
3. The current working directory should be `/home/user/IITB-GNR650-ADLCV` from where the code `main.py` need to be run.

## Results

After running all configurations, the results should be similar to:

| Configuration                  | ZSL Accuracy (%) |
|--------------------------------|------------------|
| ResNet-50 + Word2Vec           | 19.95           |
| ResNet-50 + FastText           | 35.55           |
| ViT Base + Word2Vec            | 23.03           |
| **ViT Base + FastText**        | **40.23**       |

These results demonstrate that the ViT Base with FastText configuration provides the highest ZSL accuracy.
