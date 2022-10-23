import torch
import random
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import re
import json
import os
random.seed(10)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def read_dataset(path, class_num):
    with open(path, 'r') as rf:
        texts = []
        labels = []
        dataset = rf.read().split('\n')
        for data in dataset:
            if data.strip() == '':
                continue
            text, label = data.split('\t')
            text = text.strip()
            label = int(label.strip())
            label = 0 if label == class_num else label
            texts.append(text)
            labels.append(label)
    return texts, labels

# def read_imdb_split(split_dir):
#     split_dir = Path(split_dir)
#     texts = []
#     labels = []
#     for label_dir in ["pos", "neg"]:
#         for text_file in (split_dir/label_dir).iterdir():
#             texts.append(text_file.read_text())
#             labels.append(0 if label_dir == "neg" else 1)

#     return texts, labels

def read_imdb(data_dir):
    with open(data_dir, 'rb') as f:
        data = pickle.load(data_dir)
    return data

def sampling_dataset(dataset, class_num, sample_num):
    sampled_data = []
    random.shuffle(dataset)
    stats = {label: 0 for label in range(class_num)} # ？
    while True:
        if len(sampled_data) == class_num * sample_num:
            break
        data = dataset.pop(0)
        text, label = data
        if stats[label] >= sample_num:
            dataset.append(data)
        else:
            sampled_data.append(data)
    return sampled_data, dataset

def data_sampling(dataset, sample_rate):
    # stat
    labels = [data[1] for data in dataset]
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    print("For unsampled data, n_pos:{}; n_neg:{}".format(n_pos,n_neg))

    sampled_data = []
    sampling_num = len(dataset) * sample_rate
    # sampled_data = random.choices(dataset, k = sample_num)  don't do this, because we wanna remove the corresponding elements
    random.shuffle(dataset)
    while True:
        if len(sampled_data) == sampling_num:
            break
        data = dataset.pop(0)
        sampled_data.append(data)

    sampled_labels = [data[1] for data in sampled_data]
    n_poss = sum(1 for l in labels if l == 1)
    n_negs = sum(1 for l in labels if l == 0)
    print("For sampled data, n_pos:{}; n_neg:{}".format(n_poss,n_negs))
    return sampled_data, dataset

def data_sampling_v2(dataset, sample_rate, ratio):
    '''
        sample_rate: n_labeled / n_unlabeled
        ratio: n_train / n_dev
    '''
    labels = [data[1] for data in dataset]
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    print("For unsampled data, n_pos:{}; n_neg:{}".format(n_pos,n_neg))

    sampled_data_train, sampled_data_dev = [], []
    sampling_num_train = len(dataset) * sample_rate
    sampling_num_dev = len(dataset) * sample_rate * ratio
    # sampled_data = random.choices(dataset, k = sample_num)  don't do this, because we wanna remove the corresponding elements
    random.shuffle(dataset)
    while True:
        if len(sampled_data_train) == sampling_num_train:
            break
        data = dataset.pop(0)
        sampled_data_train.append(data)
    while True:
        if len(sampled_data_dev) == sampling_num_dev:
            break
        data = dataset.pop(0)
        sampled_data_dev.append(data)
    
    sampled_labels = [data[1] for data in sampled_data_train]
    n_poss = sum(1 for l in sampled_labels if l == 1)
    n_negs = sum(1 for l in sampled_labels if l == 0)
    print("For sampled train data, n_pos:{}; n_neg:{}".format(n_poss,n_negs))

    sampled_labels = [data[1] for data in sampled_data_dev]
    n_poss = sum(1 for l in sampled_labels if l == 1)
    n_negs = sum(1 for l in sampled_labels if l == 0)
    print("For sampled dev data, n_pos:{}; n_neg:{}".format(n_poss,n_negs))
    return sampled_data_train, sampled_data_dev, dataset

