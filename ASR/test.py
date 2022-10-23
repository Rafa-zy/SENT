import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import time
import math

import yaml
import logging

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.speechxl import Transformer
from data import *
from util import *
from metric import *

#### For this file, change varaible:load_paths to specify which model you are testing#####

config_path = "/mfs/wangzhihao/research/noisy_label_higher/config.yml"

with open(config_path) as f:
    C = yaml.load(f, Loader=yaml.SafeLoader)

##########
# logger
##########
work_dir = "./"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_path = os.path.join(work_dir, 'test_log.txt')
fh = logging.FileHandler(log_path, mode='w')
ch = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

##########
# data
##########
device = torch.device('cuda')

vocab_dict, vocab_list = load_vocabulary(C['vocab_path'])


test_manifest, test_duration = read_manifest(
    mode='dev',
    manifest_path=C['test_manifest'],
    max_duration=C['max_duration'],
    min_duration=C['min_duration'], 
    max_text_len=C['max_text_len']
)

test_dataset = DatasetXL(
    manifest=test_manifest,
    vocab_dict=vocab_dict,
    vocab_list=vocab_list,
    sample_rate=C['target_sample_rate'], 
    num_concat=C['concat_size'],
    n_fft=C['n_fft'], 
    win_len=C['win_len'], 
    hop_len=C['hop_len'], 
    n_mels=C['n_mels']
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_sampler=BatchSamplerXL(
        data_source=range(len(test_dataset)),
        durations=test_duration,
        mode='dev',
        max_bsz=C['batch_size'],
        max_batch_duration=C['batch_size_in_s2'],
        shuffle=False),
    collate_fn=customize_collate_fn,
    num_workers=C['num_proc_data'],
    pin_memory=True
)
    
single_char_eos_id = 0
single_char_vocab_list = vocab_list


def convert_str(t):
    s = ''
    for i in range(t.size(0)):
        cur_char = t[i].item()
        if cur_char == single_char_eos_id:
            break
        s += single_char_vocab_list[cur_char]
    return s

punct = set(u''':!),.:;?]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/+【】@#.—.~-～''')

def remove_punctuation_from_sentence(s, lang="cn"):
    if lang == "cn":
        s = "".join(s.split())
    new_sentence = []
    for char in s:
        if char not in punct:
            new_sentence.append(char)
    return "".join(new_sentence)

def get_cer(target, prediction):
    prediction_str = convert_str(prediction)
    target_str = convert_str(target)

    # logger.info('target     | ' + target_str + '\nprediction | ' + prediction_str)

    # target_str = remove_punctuation_from_sentence(target_str)
    # prediction_str = remove_punctuation_from_sentence(prediction_str)

    return char_errors(target_str, prediction_str)


##########
# model
##########
model = Transformer(test_dataset.get_vocab_size(), C['enc_n_layer'], C['dec_n_layer'], C['n_head'],
                    C['d_model'], C['d_head'], C['d_inner'], C['dropout'], C['dropatt'],
                    C['tie_weight'], C['clamp_len'], C['chunk_size'],
                    C['n_mels'] * C['concat_size'])
load_paths = [
    #### add your trained model path here
]
for load_path in load_paths:
    print(load_path)
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)

    ##########
    # test
    ##########
    model.eval()
    total_err, total_len = 0, 0

    start_time = time.time()
    with torch.no_grad():
        for src, _, _, enc_mask, _,_,tgt_input,tgt_output,is_padding in test_dataloader:
            src = src.to(device=device, non_blocking=True)
            tgt_input = tgt_input.to(device=device, non_blocking=True)
            enc_mask = enc_mask.to(device=device, non_blocking=True)
            ori_text = tgt_input[1:][:]
            
            prediction, probs,_ = model.decode(src, enc_mask, single_char_eos_id, C['beam_size'], C['max_decode_len'])

            for i in range(prediction.size(1)):
                cur_err, cur_len = get_cer(ori_text[:, i], prediction[:, i])
                total_err += cur_err
                total_len += cur_len

    cer = total_err / total_len
    logger.info('FINAL CER: ' + str(cer))
    logger.info('TOTAL DECODE TIME: ' + str(time.time() - start_time))