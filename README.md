# KG^2: Learning to Reason Science Exam Questions with Contextual Knowledge Graph Embeddings

This repository is the PyTorch implementation for [KG2 paper](https://arxiv.org/abs/1805.12393) (Yuyu Zhang et. al. 2018).

For the dataset, [here](https://drive.google.com/file/d/1BZu8KIVJWxa_ppk7yk-Uch0dT8vbjp_o/view?usp=sharing) we provide the converted graph dictionary, token2idx dictionary and word matrix for all ARC-Dataset. Please create a `data` folder in this directory, download the dataset and unzip it in `data` folder.

To train the model, please run `python train.py`. The configuration is in `config.yaml`. 

To evaluate the model, please run `python evaluation.py --model_file {path to your model file}`. It will return the test accuracy. 