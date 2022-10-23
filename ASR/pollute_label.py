import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import math
import random

import yaml
import logging
import numpy as np

from collections import Counter

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import higher
from model.speechxl import Transformer
from data import *
from util import *
from metric import *
from copy import deepcopy
# torch.autograd.set_detect_anomaly(True)

config_path = "./config.yml"
augmentation_config = './augmentation.config'
with open(config_path) as f:
    C = yaml.load(f, Loader=yaml.SafeLoader)

C['read_hist_labels'] = False

random.seed(C['random_seed'])
np.random.seed(C['random_seed'])
torch.manual_seed(C['random_seed'])
torch.cuda.manual_seed_all(C['random_seed'])

#### For this file, change load_path to load pretrained model.  Change noise_manifest path in config.yml if you are polluting the training set. Change dev_manifest path in config.yml if you are polluting the development set (noise transfer). #####

##########
# data
##########
device = torch.device('cuda')

vocab_dict, vocab_list = load_vocabulary(C['vocab_path'])

noise_manifest, noise_duration = read_manifest(
    mode='train',
    manifest_path=C['noise_manifest'],
    max_duration=C['max_duration'],
    min_duration=C['min_duration'], 
    max_text_len=C['max_text_len'],
)

noise_dataset = DatasetXL(
    manifest=noise_manifest,
    vocab_dict=vocab_dict,
    vocab_list=vocab_list,
    sample_rate=C['target_sample_rate'], 
    num_concat=C['concat_size'],
    n_fft=C['n_fft'], 
    win_len=C['win_len'], 
    hop_len=C['hop_len'], 
    n_mels=C['n_mels'],
    augmentation_config=augmentation_config
)

dev_manifest, dev_duration = read_manifest(
    mode='dev',
    manifest_path=C['dev_manifest'],
    max_duration=C['max_duration'],
    min_duration=C['min_duration'], 
    max_text_len=C['max_text_len']
)

dev_dataset = DatasetXL(
    manifest=dev_manifest,
    vocab_dict=vocab_dict,
    vocab_list=vocab_list,
    sample_rate=C['target_sample_rate'], 
    num_concat=C['concat_size'],
    n_fft=C['n_fft'], 
    win_len=C['win_len'], 
    hop_len=C['hop_len'], 
    n_mels=C['n_mels']
)

dev_dataloader = DataLoader(
    dataset=dev_dataset,
    batch_sampler=BatchSamplerXL(
        data_source=range(len(dev_dataset)),
        durations=dev_duration,
        mode='dev',
        max_bsz=C['batch_size'],
        max_batch_duration=C['batch_size_in_s2'],
        shuffle=False),
    collate_fn=customize_collate_fn,
    num_workers=C['num_proc_data'],
    pin_memory=True
)

##########
# model
##########

model = Transformer(noise_dataset.get_vocab_size(), C['enc_n_layer'], C['dec_n_layer'], C['n_head'],
                        C['d_model'], C['d_head'], C['d_inner'], C['dropout'], C['dropatt'],
                        C['tie_weight'], C['clamp_len'], C['chunk_size'],
                        C['n_mels'] * C['concat_size'], C['label_smooth'])

model = model.to(device)



def init_weight(weight):
    nn.init.normal_(weight, 0.0, 0.02)

def init_bias(bias, init=0.0):
    nn.init.constant_(bias, init)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = C['dropout']

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = C['dropatt']

model.apply(weights_init)
for name, param in model.named_parameters():
    if 'r_w_bias' in name or 'r_r_bias' in name:
        init_weight(param)
model.word_emb.apply(weights_init)


train_step = 0
best_val_score = None

res = {}
load_path = "/mfs/wangzhihao/research/noisy_label_higher/aishell1_worse/46.0_ckpt/model.pt"
model.load_state_dict(torch.load(load_path))

##########
# test function
##########
def get_cer(target, prediction):
    prediction_str = convert_str(prediction)
    target_str = convert_str(target)
    return char_errors(target_str, prediction_str)

def convert_str(t):
    s = ''
    for i in range(t.size(0)):
        cur_char = t[i].item()
        if cur_char == 0:
            break
        s += vocab_list[cur_char]
    return s


res = {}
model.eval()
total_err, total_len = 0, 0
eval_start_time = time.time()

with torch.no_grad():
    for src, _, _, enc_mask, _,dev_inds,tgt_input,tgt_output,is_padding in dev_dataloader:
        src = src.to(device=device, non_blocking=True)
        tgt_input = tgt_input.to(device=device, non_blocking=True)
        tgt_output = tgt_output.to(device=device, non_blocking=True)
        enc_mask = enc_mask.to(device=device, non_blocking=True)
        is_padding = is_padding.to(device=device, non_blocking=True)
        ori_text = tgt_input[1:]
        prediction, probs,_ = model.decode(src, enc_mask, 0, 1, C['max_decode_len'])
        for i in range(prediction.size(1)):
            predict_str = convert_str(prediction[:, i])
            res[dev_inds[i]] = predict_str
            cur_err, cur_len = get_cer(ori_text[:, i], prediction[:, i])
            total_err += cur_err
            total_len += cur_len
    cer = (total_err / total_len) * 100.0
    log_str = 'Eval {:d} at Step {:d} | Finish within {:.2f}s | CER {:.2f}%'.format(
        train_step // C['eval_interval'] - 1, train_step, time.time() - eval_start_time, cer)
    print(log_str)

OUTPUT_DIR = "/mfs/wangzhihao/research/aishell_dev_worsemini_50percent.json"
with open(C['dev_manifest']) as ori, open(OUTPUT_DIR,'w') as f:
    for line in ori.readlines():
        info = json.loads(line)
        index = int(info['index'])
        info['text'] = res[index]
        json.dump(info,f,ensure_ascii=False)
        f.write("\n")
 