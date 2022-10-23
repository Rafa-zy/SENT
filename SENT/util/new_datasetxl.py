import os
import pickle
import re
import sys
from typing import List, Iterable
from pathlib import Path
from numpy.lib.twodim_base import mask_indices
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
import numpy as np
import joblib
import pickle
import torch
from torch.utils.data import Dataset

# EOS = 0
# UNK = 1

class DatasetXL(Dataset):
    def __init__(self,data,label_map):
        self._data = deepcopy(data)
        self._label_map = label_map
        self._label_map_reverses = {}
        for k,v in self._label_map.items():
            self._label_map_reverses[v] = k
        # self._read(manifest)
        
            
    def _read(self,manifest):
        with open(manifest,'rb') as f:
            self._data = pickle.load(f)
                         
    def _update_target(self,inds,new_target):
        new_inds = inds
        if torch.is_tensor(inds):
            new_inds = inds.cpu().tolist()
        
        for i,ind in enumerate(new_inds):
            self._data[ind]['target'] = self._label_map_reverses[new_target[i]]
    def __count_balance__(self):
        train_labels = []
        data = self._data
        for key, value in data.items():
            train_labels.append(value['original_target'])
        c = Counter(train_labels)
        return c
    
    def __getitem__(self, index):
        data = self._data[index]
        src = data['src']
        ind = int(data['index'])
        tag = self._label_map[data['target'].split()[0]]
        tag_ori = self._label_map[data['original_target'].split()[0]]
        return src, tag, ind, tag_ori

    def __len__(self):
        return len(self._data)