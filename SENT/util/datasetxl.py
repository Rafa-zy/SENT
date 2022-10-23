import os
import pickle
import re
import sys
from typing import List, Iterable
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
# from allennlp.data.tokenizers import Token
import csv
from collections import Counter
import random
from copy import deepcopy

import random
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer,DistilBertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np

# EOS = 0
# UNK = 1

TOKENIZER_bert = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER_distill = AutoTokenizer.from_pretrained("distilbert-base-uncased")
TOKENIZER_distill2 = AutoTokenizer.from_pretrained("distilroberta-base")

class DatasetXL(Dataset):
    def __init__(self, data,label_map,tokenizer=TOKENIZER_bert):
        self._label_map = label_map
        self._data = data
        self._label_map_reverses = {}
        self._tokenizer = tokenizer
        for k,v in self._label_map.items():
            self._label_map_reverses[v] = k
        # self._read(self._dir)

        
    def _read(self,manifest):
        with open(manifest,'r',encoding='utf8') as f:
            for line in f.readlines():
                info = json.loads(line.strip())
                tmp = self._tokenizer._tokenize(info['src'])
                info['len'] = len(tmp)
                self._data[int(info["index"])] = info

    def __count_balance__(self):
        train_labels = []
        data = self._data
        for key, value in data.items():
            train_labels.append(value['original_target'])
        c = Counter(train_labels)
        return c
                         
    def _update_target(self,inds,new_target):
        new_inds = inds
        if torch.is_tensor(inds):
            new_inds = inds.cpu().tolist()
        
        for i,ind in enumerate(new_inds):
            self._data[ind]['target'] = self._label_map_reverses[new_target[i]]
    
    def __getitem__(self, index):
        data = self._data[index]
        src = data['src']
        ind = int(data['index'])
        tag = self._label_map[data['target'].split()[0]]
        tag_ori = self._label_map[data['original_target'].split()[0]]
        return src, tag, ind, tag_ori

    def __len__(self):
        return len(self._data)

class BatchSamplerXL(Sampler):
    def __init__(self, data_source, mode, max_bsz, max_token_len, shuffle=True,num_groups = 5):
        self._data_source = data_source
        self._mode = mode
        self._max_bsz = max_bsz
        self._max_token_len = max_token_len
        self.shuffle = shuffle
        self.first_shuffle = False
        self._num_groups = num_groups
        
    def __iter__(self):
        # data = list(zip(self._data_source, self._durations))
        index_source = list(self._data_source._data.keys())
        lens_source = [info['len'] for info in self._data_source._data.values()]
        data_raw = np.array(list(zip(index_source, lens_source)))
        
        if self._mode == "train" and self.shuffle:
            data_groups = np.array_split(data_raw, self._num_groups)
        else:
            data_groups = np.array_split(data_raw, 1)
        end_of_batch = []
        batches = []
        cur_eob = []
        cur_batch = []
        cur_token_len = 0.0
        for dg in data_groups:
            group = dg.tolist()
            group.sort(key=lambda x:x[1], reverse=False)
            for line in dg:
                token_len = line[1]
                if cur_token_len + token_len > self._max_token_len and len(cur_batch) > 0 or len(cur_batch) == self._max_bsz:
                    cur_eob[-1] = True
                    end_of_batch.append(cur_eob)
                    batches.append(cur_batch)
                    cur_batch = []
                    cur_eob = []
                    cur_token_len = 0.0
                cur_batch.append(line)
                cur_eob.append(False)
                cur_token_len += token_len
        if len(cur_batch) > 0:
            cur_eob[-1] = True
            end_of_batch.append(cur_eob)
            batches.append(cur_batch)
        del cur_batch, cur_token_len, cur_eob
        batch_with_eob = list(zip(batches, end_of_batch))
        if self._mode == 'train' and (not self.first_shuffle or self.shuffle):
            random.shuffle(batch_with_eob)
            self.first_shuffle = True
        data = []
        end_of_batch = []
        data_, end_of_batch_ = zip(*batch_with_eob)
        for d, eob in zip(data_, end_of_batch_):
            data.extend(d)
            end_of_batch.extend(eob)
        del data_, end_of_batch_

        data_source, _ = zip(*data)

        batch = []
        for idx, flag in zip(data_source, end_of_batch):
            batch.append(int(idx))
            if flag:
                yield batch
                batch = []

def customize_collate_fn(data_batch):
    src_texts, texts,inds,texts_ori = zip(*data_batch)
    src = TOKENIZER_bert(src_texts, truncation=True, padding=True)
    return src, torch.tensor(texts), torch.tensor(inds),torch.tensor(texts_ori)


def customize_collate_fn_distill(data_batch):
    src_texts, texts,inds,texts_ori = zip(*data_batch)
    src = TOKENIZER_distill(src_texts, truncation=True, padding=True)
    return src, torch.tensor(texts), torch.tensor(inds),torch.tensor(texts_ori)

def customize_collate_fn_distill2(data_batch):
    src_texts, texts,inds,texts_ori = zip(*data_batch)
    src = TOKENIZER_distill2(src_texts, truncation=True, padding=True)
    return src, torch.tensor(texts), torch.tensor(inds),torch.tensor(texts_ori)

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = {"pos":[],"neg":[]}
    num = 0
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts[label_dir].append(text_file.read_text())
    return texts

def sampling_dataset(dataset, sample_ratio):
    sampled_data = {}
    remained_data = {}
    for key,values in dataset.items():
        sample_num = int(len(values) * sample_ratio)
        sampled_data[key] = deepcopy(values[:sample_num])
        remained_data[key] = deepcopy(values[sample_num:])
    
    return sampled_data,remained_data

'''
    9/29 We have decided to first process data into the 'dict' style.
'''

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = {"pos":[],"neg":[]}
    num = 0
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts[label_dir].append(text_file.read_text())
    return texts

LABEL_DICT_SMS = {"ham": 0, "spam": 1}
default_path_sms = "/home/linzongyu/self-training/SMS/"
default_path_trec = "/home/linzongyu/self-training/TREC/"
default_path_youtube = "/home/linzongyu/self-training/youtube/"
def load_data_SMS(mode,path=default_path_sms):
    texts = {"spam":[],"ham":[]}
    data = pd.read_csv(path + mode + ".csv", encoding='latin1')
    for tup in data.itertuples():
        sentence = getattr(tup,"v2").lower().strip()
        label = getattr(tup,"v1")
        texts[label].append(sentence)
    return texts, len(data)

def load_data_SMS_train(path=default_path_sms):
    rule_dir = default_path_sms + 'rule.txt'
    with open(rule_dir, 'r', encoding='latin1') as f:
        lines =  f.readlines()
        labels, texts = [], []
        mapping = {}
        for line in lines:
            good_train_label, good_train_text = line.strip().split('\t')[0], line.strip().split('\t')[2].lower()
            labels.append(good_train_label)
            texts.append(good_train_text)
            mapping[good_train_text] = good_train_label
    data = {'ham':[],'spam':[]}
    for t in set(texts):
        label = mapping[t]
        data[label].append(t)
    return data, len(set(texts))


LABEL_DICT_TREC_whole = {"DESCRIPTION": 0, "ENTITY": 1, "HUMAN": 2, "ABBREVIATION": 3, "LOCATION": 4, "NUMERIC": 5}
LABEL_DICT_TREC = {"DESC": 0, "ENTY": 1, "HUM": 2, "ABBR": 3, "LOC": 4, "NUM": 5}

def load_data_TREC(mode, path=default_path_trec):
    label_map = {"DESC": "DESCRIPTION", "ENTY": "ENTITY", "HUM": "HUMAN", "ABBR": "ABBREVIATION", "LOC": "LOCATION",
           "NUM": "NUMERIC"}
    texts = {key:[] for key,value in LABEL_DICT_TREC.items()}
    len_data = 0
    with open(path + mode + '.txt', 'r', encoding='latin1') as f:
        for line in f:
            #label = LABEL_DICT_TREC[label_map[line.split()[0].split(":")[0]]]
            label = line.split()[0].split(":")[0]
            if mode == "test":
                sentence = (" ".join(line.split()[1:]))
            else:
                sentence = (" ".join(line.split(":")[1:])).lower().strip()
            texts[label].append(sentence)
            len_data += 1
    return texts, len_data

def load_data_TREC_train(path=default_path_trec):
    label_map = {"DESC": "DESCRIPTION", "ENTY": "ENTITY", "HUM": "HUMAN", "ABBR": "ABBREVIATION", "LOC": "LOCATION",
           "NUM": "NUMERIC"}
    label_map_rev = {key: value for value, key in label_map.items()}
    rule_dir = default_path_trec + 'rule.txt'
    with open(rule_dir, 'r',encoding='latin1') as f:
        lines =  f.readlines()
        labels, texts = [], []
        mapping = {}
        for line in lines:
            good_train_label, good_train_text = line.strip().split('\t')[0], line.strip().split('\t')[2].lower()
            labels.append(good_train_label)
            texts.append(good_train_text)
            mapping[good_train_text] = good_train_label
    data = {key:[] for key,value in LABEL_DICT_TREC.items()}
    for t in set(texts):
        label = label_map_rev[mapping[t]]
        data[label].append(t)
    return data, len(set(texts))


LABEL_DICT_YOUTUBE = {"ham": 0, "spam": 1}
def load_data_YOUTUBE(mode, path=default_path_youtube):
    label_map_rev = {key: value for value, key in LABEL_DICT_YOUTUBE.items()}
    texts = {key:[] for key,value in LABEL_DICT_YOUTUBE.items()}
    len_data = 0
    with open(path + mode + '.txt', 'r', encoding='utf8') as f:
        for line in f:
            #label = LABEL_DICT_TREC[label_map[line.split()[0].split(":")[0]]]
            info = json.loads(line.strip())
            label = int(info['label'])
            label_after_map = label_map_rev[label]
            sentence = info['text']
            texts[label_after_map].append(sentence)
            len_data += 1
    return texts, len_data

LABEL_DICT_AG = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
def load_data_AG(mode, path):
    """Generate AG News examples."""
    label_ag_rev = {key: value for value, key in LABEL_DICT_AG.items()}
    with open(path + mode + '.csv', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        len_data = 0
        data = {key:[] for key,value in LABEL_DICT_AG.items()}
        for id_, row in enumerate(csv_reader):
            label, title, description = row
            # Original labels are [1, 2, 3, 4] ->
            #                   ['World', 'Sports', 'Business', 'Sci/Tech']
            # Re-map to [0, 1, 2, 3].
            label = int(label) - 1
            text = " ".join((title, description))
            data[label_ag_rev[label]].append(text)
            len_data += 1
    return data, len_data
            