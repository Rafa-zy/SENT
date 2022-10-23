import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import sys
import time
import math
import random

import yaml
import logging
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import higher

from model.speechxl import Transformer
from data import *
from util import *
from metric import *

#### Change noise_manifest path, dev_manifest path in config.yml to specify the datasets. Change the pretrain_dir to specify the directory for saving models.#####


config_path = '/mfs/wangzhihao/research/noisy_label_higher/config.yml'
augmentation_config = '/mfs/wangzhihao/research/noisy_label_higher/augmentation.config'

with open(config_path) as f:
    C = yaml.load(f, Loader=yaml.SafeLoader)
if not os.path.exists(C['pretrain_dir']):
    os.makedirs(C['pretrain_dir'])
print(f"work dir is {C['pretrain_dir']}, data dir is {C['noise_manifest']}")
random.seed(C['random_seed'])
np.random.seed(C['random_seed'])
torch.manual_seed(C['random_seed'])
torch.cuda.manual_seed_all(C['random_seed'])

##########
# logger
##########
work_dir = C['pretrain_dir']
log_path = os.path.join(work_dir, 'log.txt')
log_path2 = os.path.join(work_dir, 'step_metaloss.txt')
logger = logging.getLogger()

logger.setLevel(logging.INFO)

fh = logging.FileHandler(log_path, mode='w')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)



##########
# data
##########
device = torch.device('cuda')

vocab_dict, vocab_list = load_vocabulary(C['vocab_path'])

# noise_manifest, noise_eob = read_manifest(
#     mode='train',
#     manifest_path=C['noise_manifest'],
#     max_bsz=C['batch_size'],
#     max_duration=C['max_duration'],
#     min_duration=C['min_duration'], 
#     max_text_len=C['max_text_len'],
#     max_batch_duration=C['batch_size_in_s2']
# )

noise_manifest, noise_duration = read_manifest(
    mode='train',
    manifest_path=C['noise_manifest'],
    max_duration=C['max_duration'],
    min_duration=C['min_duration'], 
    max_text_len=C['max_text_len'],
)

# noise_dataset = DatasetXL(
#     manifest=noise_manifest,
#     vocab_dict=vocab_dict,
#     vocab_list=vocab_list,
#     sample_rate=C['target_sample_rate'], 
#     num_concat=C['concat_size'],
#     n_fft=C['n_fft'], 
#     win_len=C['win_len'], 
#     hop_len=C['hop_len'], 
#     n_mels=C['n_mels'],
#     # augmentation_config=augmentation_config
# )
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
noise_dataloader = DataLoader(
    dataset=noise_dataset,
    batch_sampler=BatchSamplerXL(
        data_source=range(len(noise_manifest)),
        durations=noise_duration,
        mode='train',
        max_bsz=C['batch_size'],
        max_batch_duration=C['batch_size_in_s2'],
        shuffle=C['shuffle']),
    collate_fn=customize_collate_fn,
    num_workers=C['num_proc_data'],
    pin_memory=True,
)

# clean_manifest, clean_duration = read_manifest(
#     mode='train',
#     manifest_path=C['clean_manifest'],
#     max_duration=C['max_duration'],
#     min_duration=C['min_duration'], 
#     max_text_len=C['max_text_len'],
# )

# clean_dataset = DatasetXL(
#     manifest=clean_manifest,
#     vocab_dict=vocab_dict,
#     vocab_list=vocab_list,
#     sample_rate=C['target_sample_rate'], 
#     num_concat=C['concat_size'],
#     n_fft=C['n_fft'], 
#     win_len=C['win_len'], 
#     hop_len=C['hop_len'], 
#     n_mels=C['n_mels']
# )

# clean_dataloader = DataLoader(
#     dataset=clean_dataset,
#     batch_sampler=BatchSamplerXL(
#         data_source=range(len(clean_dataset)),
#         durations=clean_duration,
#         mode='dev',
#         max_bsz=C['batch_size'],
#         max_batch_duration=C['batch_size_in_s2'],
#         shuffle=C['shuffle']),
#     collate_fn=customize_collate_fn,
#     num_workers=C['num_proc_data'],
#     pin_memory=True
# )

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
        shuffle=C['shuffle']),
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

##########
# rbf meta
##########
# center = torch.tensor([-8.5], device=device).requires_grad_(True)
# sigma = 10 * torch.ones(1, device=device).requires_grad_(True)
center = torch.tensor([0.5], device=device).requires_grad_(True)
sigma = torch.tensor([3.], device=device).requires_grad_(True)

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

optimizer = torch.optim.Adam(model.parameters(), lr=C['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, C['max_step']-C['warmup_step'], eta_min=C['lr_min'])
train_step = 0
best_val_score = None
#for burn-in
burn = C['burn_in']
burn_steps = C['burn_in_steps']
final_burn = C['final_burn_drop']
start_burn = C['start_burn_drop']

if C['resume_training']:
    best_val_score = torch.load(os.path.join(C['resume_dir'], 'best_cer.pt'))
    model.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'model.pt')))
    model.apply(update_dropout)
    model.apply(update_dropatt)
    center = torch.load(os.path.join(C['resume_dir'], 'center.pt'))
    sigma = torch.load(os.path.join(C['resume_dir'], 'sigma.pt'))
    if sigma[0] == 0:
        sigma = -1 * torch.ones(1, device=device).requires_grad_(True)
    if C['re_training']:
        optimizer.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'opt.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'sch.pt')))
        train_step = torch.load(os.path.join(C['resume_dir'], 'step.pt'))
        

##########
# info
##########
C['asr_param'] = sum([p.nelement() for p in model.parameters()])
logger.info('=' * 40)
for c in C:
    logger.info('     - {} : {}'.format(c, C[c]))
logger.info('=' * 40)

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

class BurnDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dropout=0.5):
        if not self.training or not dropout:
            return input
        m = input.data.new(1, input.size(1), input.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False)
        mask = mask.expand_as(input)
        return mask * input

##########
# train
##########
def grad_clipping(grads, max_norm=0.25, norm_type=2.0):
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.detach().mul_(clip_coef.to(g.device))
    return grads

# clean_data_iter = iter(clean_dataloader)
noise_data_iter = iter(noise_dataloader)
log_loss, log_meta_loss = 0.0, 0.0
log_start_time = time.time()

loss_log = []
weight_log = []
mu_log = []
sigma_log = []

def calc_weight(log_prob,targets,is_padding,center,sigma,token_level=False):
    token_nums = torch.sum(~is_padding)
    if token_level:
        cost = F.nll_loss(log_prob.permute(1,2,0), targets.T, reduction='none').T
        cost = cost.masked_fill(is_padding,torch.finfo(cost.dtype).max)
        min_val = torch.min(cost)
        cost = cost.masked_fill(is_padding,torch.finfo(cost.dtype).min)
        max_val = torch.max(cost)
        score = (cost - min_val) / (max_val - min_val)
        score = score * (~is_padding)
        cost = cost * (~is_padding)
    else:
        cost = F.nll_loss(log_prob.permute(1,2,0), targets.T, reduction='none').T * (~is_padding)
        cost = torch.sum(cost, dim=0) / torch.sum(~is_padding, dim=0)
        score = (cost - cost.min()) / (cost.max() - cost.min())
        # distances = (score - center).pow(2).pow(0.5) * sigma
        # weight = torch.exp(-1 * distances.pow(2))
    distances = (score - center).pow(2).pow(0.5) * sigma
    weight = torch.exp(-1 * distances.pow(2))
    if token_level:
        weight = weight * (~is_padding)
        cost = cost / token_nums #scale loss
    return weight,cost,score

for epoch in range(0, 1000):
    model.train()
    batch_idx = 0
    need_update = False
    one_epoch_finish = False
    while not one_epoch_finish:
        if train_step >= C['max_step']:
                logging('-' * 100)
                logging('End of training')
                os._exit(0)
        if C['warmup_step'] > 0 and train_step < C['warmup_step']:
            optimizer.param_groups[0]['lr'] = C['lr'] * train_step / C['warmup_step']
        if (C['pre_train'] and train_step >= C['pre_train_steps']) or not C['pre_train']: 
            need_update = True
        try:
            src, tgt_input, tgt_output, enc_mask, is_padding,inds,tgt_input_ori,tgt_output_ori,is_padding_ori = next(noise_data_iter)
        except StopIteration:
            noise_data_iter = iter(noise_dataloader)
            one_epoch_finish = True
            break
        batch_idx += 1
        src = src.to(device=device, non_blocking=True)
        tgt_input = tgt_input.to(device=device, non_blocking=True)
        tgt_output = tgt_output.to(device=device, non_blocking=True)
        enc_mask = enc_mask.to(device=device, non_blocking=True)
        is_padding = is_padding.to(device=device, non_blocking=True)
        optimizer.zero_grad()
        logit,_,_,_ = model(src, tgt_input, enc_mask)
        log_prob = F.log_softmax(logit, dim=-1)
        cost = F.nll_loss(log_prob.permute(1,2,0), tgt_output.T, reduction='none').T * (~is_padding)
        loss = cost.sum() / (~is_padding).sum()
        log_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        if train_step >= C['warmup_step']:
            scheduler.step()
        train_step += 1

        if train_step % C['log_interval'] == 0:
            logger.info('mu:'+str(center.item())+'|sigma:'+str(sigma.item()))
            elapsed = time.time() - log_start_time
            log_str = 'Epoch {:d}, Step {:d}, Batch {:d} | Speed {:.2f}ms/it, LR {:3g} | Loss {:.4f}, Meta Loss {:.4f}'.format(
                        epoch, train_step, batch_idx,
                        elapsed * 1000 / C['log_interval'],
                        optimizer.param_groups[0]['lr'],
                        log_loss / C['log_interval'], log_meta_loss / C['log_interval'])
            logger.info(log_str)
            log_loss, log_meta_loss = 0.0, 0.0
            log_start_time = time.time()

        if train_step % C['dist_log_interval'] == 0:
            if len(loss_log) == 0:
                logger.info(f"loss_mean: {0},loss_std: {0}")
                logger.info(f"weight_mean: {0},weight_std: {0}")
                logger.info(f"mu_mean: {0},sigma_mean: {0}")
            else:   
                loss_log = np.array(loss_log)
                mu_log = np.array(mu_log)
                sigma_log = np.array(sigma_log)
                weight_log = np.array(weight_log)
                logger.info(f"loss_mean: {loss_log.mean():.4f},loss_std: {loss_log.std():.4f}")
                logger.info(f"weight_mean: {weight_log.mean():.4f},weight_std: {weight_log.std():.4f}")
                logger.info(f"mu_mean: {mu_log.mean():.4f},sigma_mean: {sigma_log.mean():.4f}")
            loss_log = []
            weight_log = []
            mu_log = []
            sigma_log = []

        if train_step % C['eval_interval'] == 0:
            order = str(train_step / C['eval_interval'])
            savedir = os.path.join(work_dir, f"{order}_ckpt/")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            # eval loop
            model.eval()
            total_err, total_len = 0, 0
            eval_start_time = time.time()
            with torch.no_grad():
                for src, _, _, enc_mask, _,_,tgt_input,tgt_output,is_padding in dev_dataloader:
                    src = src.to(device=device, non_blocking=True)
                    tgt_input = tgt_input.to(device=device, non_blocking=True)
                    tgt_output = tgt_output.to(device=device, non_blocking=True)
                    enc_mask = enc_mask.to(device=device, non_blocking=True)
                    is_padding = is_padding.to(device=device, non_blocking=True)
                    ori_text = tgt_input[1:][:]
                    
                    prediction, probs,_ = model.decode(src, enc_mask, 0, 1, C['max_decode_len'])

                    for i in range(prediction.size(1)):
                        cur_err, cur_len = get_cer(ori_text[:, i], prediction[:, i])
                        total_err += cur_err
                        total_len += cur_len
                cer = (total_err / total_len) * 100.0
                log_str = 'Eval {:d} at Step {:d} | Finish within {:.2f}s | CER {:.2f}%'.format(
                    train_step // C['eval_interval'] - 1, train_step, time.time() - eval_start_time, cer)
                logger.info(log_str)
                if best_val_score is None or cer <= best_val_score:
                    torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pt'))
                    torch.save(cer, os.path.join(work_dir, 'best_cer.pt'))
                    best_val_score = cer
                    logger.info('Better CER detected, model saved...')
                torch.save(model.state_dict(), os.path.join(savedir, f'model.pt'))
                torch.save(cer, os.path.join(savedir, f'cer_{cer:.2f}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(savedir, 'opt.pt'))
                torch.save(scheduler.state_dict(), os.path.join(savedir, 'sch.pt'))
                torch.save(train_step, os.path.join(savedir, 'step.pt'))
            model.train()
            
    
