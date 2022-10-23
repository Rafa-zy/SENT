import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from model.speechxl import Transformer
from data import *
from util import *
from metric import *
from copy import deepcopy
from difflib import SequenceMatcher
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

# torch.autograd.set_detect_anomaly(True)
NOTSAME_SELECT_THRESHOLD = 0.5
SAME_SELECT_THRESHOLD = 0.5

config_path = "./config.yml"
augmentation_config = './augmentation.config'
single_char_eos_id = 0

NUMERICAL_EPS = 1e-12
with open(config_path) as f:
    C = yaml.load(f, Loader=yaml.SafeLoader)
C['read_hist_labels'] = False
if not C['min_max']:
    init_hist_track_path = "./hist_track_train_20len_originscale.npy"
    init_hist_loss_path = "./hist_losses_train_originscale.npy"    
    init_hist_track_path_dev = "./hist_track_dev_20len_originscale.npy"
    init_hist_loss_path_dev = "./hist_losses_dev_originscale.npy" 
else:
    init_hist_track_path = "./hist_track_train_20len.npy"
    init_hist_loss_path = "./hist_losses_train.npy"    
    init_hist_track_path_dev = "./hist_track_dev_20len.npy"
    init_hist_loss_path_dev = "./hist_losses_dev.npy"

if not C['search']:
    C['beta'] = 1    
if not C['use_aug']:
    augmentation_config = './augmentation_dev.config'
# if C['low_LR']:
    # C['max_step'] = 100000
work_dir = C['work_dir']
suffix = f"exp_debug_LCSafter_UpdateSelect"
if not C['debug']:
    suffix += f"_{C['loss_gamma_min']}lossgammamin_{C['max_step']}maxstep_{C['input_size']}input_{C['hidden_size']}hidsize_{C['pretrain_epochs']}prepochs_{C['act']}act_{C['max_history_len']}histlen_{C['multi_rounds']}rounds"
    if C['pre_train']:
        suffix += f"_{(C['pre_train_steps'])}pre"
        suffix += f"_{C['pre_train_optim_steps']}optimsteps"
    
    suffix += f"_{C['meta_lr']}metalr"
    suffix += f"_{C['acc_epochs']}acc"

    if C['only_different']:
        suffix += "_onlydif"
    if C['hist_accumulate']:
        suffix += "_histacc"
    if C['include_GT']:
        suffix += "_includeGT"
    if C['bnm']:
        suffix += "_bnm"
    if C['new_TF']:
        suffix += "_newTF"
    if C['min_max']:
        suffix += "_minmax"
    if C['use_aug']:
        suffix += "_useaug"
    if C['soft_select']:
        suffix += "_softselect"
    if C['self_train']:
        suffix += "_selftrain"
    if C['hidden_feature']:
        suffix += "_hidfeat"
    if C['weighted_loss']:
        suffix += "_weightedloss"
    if C['search']:
        suffix += f"_search{C['search_target']}{C['beta']}"
    if C['resume_training']:
        if C['resume_dir'][-2] in ['2','3','4','5']:
            suffix += str(int(C['resume_dir'][-2])+1)
        else:
            suffix += '2'
else:
    C['noise_manifest'] = "./aishell_fusion_miniinfer_1000.json"
    C['clean_manifest'] = "./aishell_fusion_dev_ios_doublelabel_500.json"
suffix += "/"
work_dir += suffix

if not os.path.exists(work_dir):
    os.makedirs(work_dir)
print(f"work dir is {work_dir}")
random.seed(C['random_seed'])
np.random.seed(C['random_seed'])
torch.manual_seed(C['random_seed'])
torch.cuda.manual_seed_all(C['random_seed'])

##########
# logger
##########

log_path = os.path.join(work_dir, 'log.txt')
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


clean_manifest, clean_duration = read_manifest(
    mode='train',
    manifest_path=C['dev_manifest'],
    max_duration=C['max_duration'],
    min_duration=C['min_duration'], 
    max_text_len=C['max_text_len'],
)

clean_dataset = DatasetXL(
    manifest=clean_manifest,
    vocab_dict=vocab_dict,
    vocab_list=vocab_list,
    sample_rate=C['target_sample_rate'], 
    num_concat=C['concat_size'],
    n_fft=C['n_fft'], 
    win_len=C['win_len'], 
    hop_len=C['hop_len'], 
    n_mels=C['n_mels'],
    augmentation_config=augmentation_config if C['use_aug'] else None
)

clean_dataloader = DataLoader(
    dataset=clean_dataset,
    batch_sampler=BatchSamplerXL(
        data_source=range(len(clean_dataset)),
        durations=clean_duration,
        mode='train',
        max_bsz=C['dev_batch_size'],
        max_batch_duration=C['dev_batch_size_in_s2'],
        shuffle=C['shuffle']),
    collate_fn=customize_collate_fn,
    num_workers=C['num_proc_data'],
    pin_memory=True
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
        max_bsz=C['dev_batch_size'],
        max_batch_duration=C['dev_batch_size_in_s2'],
        shuffle=C['shuffle']),
    collate_fn=customize_collate_fn,
    num_workers=C['num_proc_data'],
    pin_memory=True
)


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

##########
# model
##########
class MLP_head(nn.Module):
    def __init__(self,hid_size,input_size,output_size,act="elu",bnm=False):
        super(MLP_head, self).__init__()
        self.hidden_size = hid_size
        self.input_size = input_size
        self.output_size = output_size
        self.act_type = act
        self.bnm = MaskedBatchNorm1d(num_features = self.input_size)
        self.bnm_flag = bnm
        if self.hidden_size > 0:
            self.linear1 = nn.Linear(self.input_size,self.hidden_size)
            if self.act_type == "elu":
                self.act=nn.ELU()
            else:
                self.act = nn.ReLU()
            self.linear2 = nn.Linear(self.hidden_size,self.output_size)
        else:
            self.linear2 = nn.Linear(self.input_size,self.output_size)


    def forward(self,x,padding_mask=None,softmax=True):
        if self.bnm_flag:
            x = x.permute(1,2,0)
            mask_for_bnm = padding_mask.permute(1,2,0)
            x = self.bnm(x,mask_for_bnm)
            x = x.permute(2,0,1)
        if self.hidden_size > 0:
            x = self.linear1(x)
            x = self.act(x)
        out = self.linear2(x)
        if softmax:
            out = F.softmax(out,dim=-1)
        if padding_mask is not None:
            out = out * (padding_mask)
        return out

model = Transformer(noise_dataset.get_vocab_size(), C['enc_n_layer'], C['dec_n_layer'], C['n_head'],
                        C['d_model'], C['d_head'], C['d_inner'], C['dropout'], C['dropatt'],
                        C['tie_weight'], C['clamp_len'], C['chunk_size'],
                        C['n_mels'] * C['concat_size'], C['label_smooth'])

model = model.to(device)

model.apply(weights_init)
for name, param in model.named_parameters():
    if 'r_w_bias' in name or 'r_r_bias' in name:
        init_weight(param)

meta_model = MLP_head(C['hidden_size'],C['input_size'],C['output_size'],C['act'],C['bnm'])
meta_model = meta_model.to(device)
meta_model_cls_notsame = MLP_head(C['hidden_size'],4 +1 if C['hidden_feature'] else 4,2,C['act'],False)
meta_model_cls_same = MLP_head(C['hidden_size'],4+ 1 if C['hidden_feature'] else 4,2,C['act'],False)
meta_model_cls_notsame = meta_model_cls_notsame.to(device)
meta_model_cls_same = meta_model_cls_same.to(device)
ce_loss = torch.nn.CrossEntropyLoss(torch.tensor([0.2,0.8]),ignore_index=-1).to(device) if C['weighted_loss'] else torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
meta_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=C['lr'])
optimizer_meta = torch.optim.Adam(meta_model.parameters(), lr=C['meta_lr'])
optimizer_meta_cls_notsame = torch.optim.Adam(meta_model_cls_notsame.parameters(), lr=C['meta_lr_cls'])
optimizer_meta_cls_same = torch.optim.Adam(meta_model_cls_same.parameters(), lr=C['meta_lr_cls'])

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, C['max_step']-C['warmup_step'], eta_min=C['lr_min'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = C['lr_gamma'])

max_uncertainty = -np.log(1.0/float(len(vocab_list)))

gamma_change_total_epochs = 30
scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta, (C['max_step'] - C['warmup_step'])/10, eta_min=C['meta_lr_min'])
scheduler_meta_cls_notsame = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta_cls_notsame, (C['max_step']-C['warmup_step'])/10, eta_min=C['meta_lr_cls_min'])
scheduler_meta_cls_same = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta_cls_same,(C['max_step']-C['warmup_step'])/10, eta_min=C['meta_lr_cls_min'])

loss_gamma_step = (C['loss_gamma_max'] -  C['loss_gamma_min']) / gamma_change_total_epochs
loss_gamma = C['loss_gamma_min']
max_history_len = C['max_history_len']
max_text_len_dicts = {"worsemini":83,"miniinfer":92,"bettermini":77}
dev_max_text_len_dicts = {"worsemini":101,"miniinfer":101,"bettermini":101}
max_text_len = max_text_len_dicts[C['training_sets']]
max_lines = 120099
max_lines_dev = 14327
C['meta_lr_step_ratio'] = int(max_lines / max_lines_dev)
logger.info(f"meta_lr_step_ratio {C['meta_lr_step_ratio']}")
max_text_len_dev = dev_max_text_len_dicts[C['training_sets']]
if not (C['read_hist_labels'] or C['resume_training']):
    hist_track = np.zeros(shape=(max_lines,max_text_len,max_history_len),dtype=np.int16) #bsz * len * max_hist_len
    hist_losses = np.zeros(shape=(max_lines,max_text_len))
    hist_track_dev = np.zeros(shape=(max_lines_dev,max_text_len_dev,max_history_len),dtype=np.int16) #bsz * len * max_hist_len
    hist_losses_dev = np.zeros(shape=(max_lines_dev,max_text_len_dev))

train_step = 0
best_val_score = None


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

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = C['dropout']

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = C['dropatt']


def min_max_normalize(target,is_padding = None):
    if is_padding is not None:
        # return target * (~is_padding)
        target = target.masked_fill(is_padding,torch.finfo(target.dtype).max)
        min_val = torch.min(target)
        target = target.masked_fill(is_padding,0.)
        max_val = torch.max(target)
        normalized_target = (target - min_val.detach()) / ((max_val - min_val) + torch.finfo(target.dtype).eps).detach()
        return normalized_target * (~is_padding)
    else:
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        min_val = np.min(target)
        max_val = np.max(target)
        tmp = (target - min_val) / (max_val - min_val + NUMERICAL_EPS)
        return tmp


def calc_loss(log_prob,targets,is_padding,hist_loss=None,loss_gamma=0.9):
    #score is normalized cost
    cost = F.kl_div(log_prob.transpose(1,0), targets.transpose(1,0), reduction='none').sum(dim=-1).T
    # cost_different = cost
    cost_padded = min_max_normalize(cost,is_padding)
    score = cost_padded
    normalized_cost = score
    
    if hist_loss is not None:
        if type(hist_loss) == np.ndarray:
            hist_loss = torch.from_numpy(hist_loss).to(device)
        score = loss_gamma * hist_loss + (1-loss_gamma) * score
    return 1,cost_padded,score,normalized_cost

def calc_entropy(hist_label,cur_pred,is_padding,cur_update_ind):
    bsz,ts,_ = hist_label.shape
    hist_uncer_np = np.zeros(shape=(bsz,ts))
    instant_uncer_np = np.zeros(shape=(bsz,ts))
    for i,token_steps in enumerate(hist_label):
        for j,pred_labels in enumerate(token_steps):
            c = Counter(pred_labels)
            keys = list(c.keys())
            values = np.array(list(c.values()))
            values = values / (cur_update_ind+1)
            values = values * np.log(values)
            values = -1 * values / max_uncertainty
            pred_key = cur_pred[i][j].item()
            if not pred_key in keys:
                instant_uncer_np[i][j] = 0
            else:
                index = keys.index(pred_key)
                instant_uncer_np[i][j] = values[index]
            uncertainty = values.sum()
            hist_uncer_np[i][j] = uncertainty
    hist_uncer_tc = torch.from_numpy(hist_uncer_np).to(device)
    instant_uncer_tc = torch.from_numpy(instant_uncer_np).to(device)
    return min_max_normalize(hist_uncer_tc,is_padding),min_max_normalize(instant_uncer_tc,is_padding)

def get_major_vote_label(hist_label):
    bsz,ts,_ = hist_label.shape
    res = np.zeros(shape=(bsz,ts),dtype=np.int64)
    for i,token_steps in enumerate(hist_label):
        for j,pred_labels in enumerate(token_steps):
            c = Counter(pred_labels)
            major_vote = c.most_common(1)[0][0]
            res[i][j] = major_vote
    return res


start_epoch = C['resume_epoch'] if C['resume_training'] else 0
cur_save_id = 0
if C['read_hist_labels']:
    start_epoch = 7
if C['pre_train']:
    order = str(C['pre_train_steps'] / C['eval_interval'])
    savedir = os.path.join(C['pretrain_dir'], f"{order}_ckpt/")
    model.load_state_dict(torch.load(os.path.join(savedir, 'model.pt')))
    model.apply(update_dropout)
    model.apply(update_dropatt)

    order = str(C['pre_train_optim_steps'] / C['eval_interval'])
    savedir = os.path.join(C['pretrain_dir'], f"{order}_ckpt/")
    train_step = C['pre_train_steps']
    scheduler.step(train_step)
    print(optimizer.param_groups[0]['lr'])

if C['resume_training']:
    best_val_score = torch.load(os.path.join(C['resume_dir'], 'best_cer.pt'))
    model.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'best_model.pt')))
    model.apply(update_dropout)
    model.apply(update_dropatt)
    hist_track = np.load(os.path.join(C['resume_dir'],"hist_track.npy"),allow_pickle=True) #bsz * len * max_hist_len
    hist_track_dev = np.load(os.path.join(C['resume_dir'],"hist_track_dev.npy"),allow_pickle=True) #bsz * len * max_hist_len
    hist_losses = np.load(os.path.join(C['resume_dir'],"hist_losses.npy"),allow_pickle=True) #bsz * len  
    hist_losses_dev = np.load(os.path.join(C['resume_dir'],"hist_losses_dev.npy"),allow_pickle=True) #bsz * len  
    optimizer.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'best_opt.pt')))
    scheduler.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'best_sch.pt')))
    train_step = torch.load(os.path.join(C['resume_dir'], 'best_step.pt'))
    hist_accu_labels = np.load(os.path.join(C['resume_dir'],"hist_accu_labels.npy"),allow_pickle=True)
    hist_accu_labels = hist_accu_labels[()]
    hist_accu_labels_dev = np.load(os.path.join(C['resume_dir'],"hist_accu_labels_dev.npy"),allow_pickle=True)
    hist_accu_labels_dev = hist_accu_labels_dev[()]
    # meta_model.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'best_meta_model.pt')))
    # optimizer_meta.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'best_meta_optimizer.pt')))
    # scheduler_meta.load_state_dict(torch.load(os.path.join(C['resume_dir'], 'best_meta_scheduler.pt')))
if C['read_hist_labels'] and init_hist_track_path is not None and init_hist_loss_path is not None:        
    hist_track = np.load(init_hist_track_path,allow_pickle=True) #bsz * len * max_hist_len
    hist_losses = np.load(init_hist_loss_path,allow_pickle=True) #bsz * len 
    hist_losses_dev = np.load(init_hist_loss_path_dev,allow_pickle=True)
    hist_track_dev = np.load(init_hist_track_path_dev,allow_pickle=True)
if not C['resume_training']:
    hist_accu_labels = {}
    hist_accu_labels_dev = {}



##########
# info
##########
C['asr_param'] = sum([p.nelement() for p in model.parameters()])
logger.info('=' * 40)
for c in C:
    logger.info('     - {} : {}'.format(c, C[c]))
logger.info('=' * 40)


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

noise_data_iter = iter(noise_dataloader)
log_loss = 0.0
log_start_time = time.time()
criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

cur_loss_log = []
loss_log = []
weight_gt_log = []
weight_acc_log = []
weight_major_log = []
weight_pred_log = []


def get_metrics(ema_loss=None,cur_loss=None,hist_entropy=None,cur_entropy=None,indexes=None):
    if indexes is None:
        return 
    ema_loss_metric = None
    cur_loss_metric = None
    hist_entropy_metric = None
    cur_entropy_metric = None
    if ema_loss is not None:
        ema_loss_metric = ema_loss[indexes].detach().cpu().numpy()
    if cur_loss is not None:
        cur_loss_metric = cur_loss[indexes].detach().cpu().numpy()
    if hist_entropy is not None:
        hist_entropy_metric = hist_entropy[indexes].detach().cpu().numpy()
    if cur_entropy is not None:
        cur_entropy_metric = cur_entropy[indexes].detach().cpu().numpy()
    return ema_loss_metric,cur_loss_metric,hist_entropy_metric,cur_entropy_metric

def get_hist_accu_probs(tgt_prob,hist_preds,inds,hist_accu_probs,first_epoch=False):
    if first_epoch:#avg_gt and hist_pred
        noisy_gt_prob = tgt_prob
        hist_prob = F.one_hot(torch.from_numpy(hist_preds).long(),num_classes=len(vocab_list)).float().to(device)
        if C['include_GT']:
            hist_accu_prob = torch.cat((noisy_gt_prob[:,:,None,:],hist_prob),dim=-2)
        hist_accu_prob = hist_prob.mean(dim=-2)
    else:
        hist_accu_prob = get_history_labels(inds,hist_accu_probs).to(device) #ts,bsz,len(vocab)
    return hist_accu_prob.float()
def update_all_history(inds,cur_len,cur_update_ind,score,cur_pred,hist_accu_probs=None,new_pred_prob=None,is_padding=None,mode="dev"):
    #update hist losses
    if mode == "train":
        hist_losses[inds,:cur_len] = score.transpose(1,0).detach().clone().cpu().numpy()
        hist_track[inds,:cur_len,cur_update_ind] = cur_pred.transpose(1,0).clone().cpu().numpy().astype('int16')
    elif mode == "dev":
        hist_losses_dev[inds,:cur_len] = score.transpose(1,0).detach().clone().cpu().numpy()
        hist_track_dev[inds,:cur_len,cur_update_ind] = cur_pred.transpose(1,0).clone().cpu().numpy().astype('int16')
    else:
        raise ValueError("no other modes")
    if C["hist_accumulate"] and new_pred_prob != None and hist_accu_probs != None:
        update_history_labels(inds,hist_accu_probs,new_pred_prob,is_padding)


def pad_2_seq(pred,gt,mode="after"):
    dis_len = abs(len(pred) - len(gt))
    gt_is_padding = torch.zeros((len(gt),))
    if mode == "after":
        if len(pred) <= len(gt):
            padded_gt = gt[:len(gt)-dis_len]
            is_padding = gt_is_padding[:len(gt)-dis_len]
        else:
            pads = torch.zeros(size=(dis_len,))
            padded_gt = torch.cat((gt,pads),dim=0)
            is_padding = torch.cat((gt_is_padding,(~pads.bool()).float()),dim=0)
    elif mode =="before":
        if len(pred) <= len(gt):
            padded_gt = gt[dis_len:]
            is_padding = gt_is_padding[dis_len:]
        else:
            pads = torch.zeros(size=(dis_len,))
            padded_gt = torch.cat((pads,gt),dim=0)
            is_padding = torch.cat(((~pads.bool()).float(),gt_is_padding),dim=0)

    return padded_gt,is_padding


def get_padded_gt(tgt_output,tgt_output_ori,is_padding_pred,is_padding_ori):
    need_LCS_token_nums = []
    need_LCS = 0

    noneed_LCS = 0
    bsz = tgt_output.size(1)
    tgt_output_cpu = tgt_output.cpu()
    tgt_input_ori_cpu = tgt_output_ori.cpu()
    new_tgt_output = []
    new_tgt_output_ori = []
    new_is_padding = []
    for i in range(bsz):
        tmp_pred = tgt_output_cpu[:,i].squeeze()
        tmp_gt = tgt_input_ori_cpu[:,i].squeeze()
        tmp_pad_pred = is_padding_pred[:,i].squeeze()
        tmp_pad_gt = is_padding_ori[:,i].squeeze()
        pred_ind = len(tmp_pad_pred) - tmp_pad_pred.sum() #注意有一个EOS在最后
        gt_ind = len(tmp_pad_pred) - tmp_pad_gt.sum()
        pred = tmp_pred[:pred_ind]
        gt = tmp_gt[:gt_ind]
        match = SequenceMatcher(None,pred.tolist(),gt.tolist()).get_matching_blocks()
        match = sorted(match,key=lambda x: x.size, reverse=True)
        if match[0].size == 0: #no match
            padded_gt,is_padding = pad_2_seq(pred,gt)
            noneed_LCS += 1
        else:
            pred_start_index = match[0].a
            pred_end_index = match[0].a + match[0].size
            gt_start_index = match[0].b
            gt_end_index = match[0].b + match[0].size
            #剩下的部分（对齐前面的part，和对齐后面的part），如果GT短，就在GT的后面补0，否则就截断GT
            pred_front = pred[:pred_start_index]
            gt_front = gt[:gt_start_index]
            padded_gt_front,is_padding_front = pad_2_seq(pred_front,gt_front,mode='before')
            pred_back = pred[pred_end_index:]
            gt_back = gt[gt_end_index:]
            padded_gt_back,is_padding_back = pad_2_seq(pred_back,gt_back)
            padded_gt = torch.cat((padded_gt_front,gt[gt_start_index:gt_end_index],padded_gt_back),dim=0)
            is_padding = torch.cat((is_padding_front,torch.zeros((match[0].size,)),is_padding_back),dim=0)
            if is_padding.sum() == 0:
                noneed_LCS += 1
            else:
                need_LCS += 1
                need_LCS_token_nums.append(match[0].size)
        # new_tgt_output.append(pred)
        new_tgt_output_ori.append(padded_gt)
        new_is_padding.append(is_padding)
    # new_tgt_output = pad_sequence(new_tgt_output,padding_value=0)
    new_tgt_output_ori = pad_sequence(new_tgt_output_ori,padding_value=0)
    new_is_padding = pad_sequence(new_is_padding,padding_value=1)
    return new_tgt_output_ori.to(device).long(),new_is_padding.to(device).bool(),need_LCS,noneed_LCS,need_LCS_token_nums


def update_meta_models(cur_update_ind,cur_fetch_ind,update_history=False,first_epoch=False,only_infer=False):
    '''
    update_history means 是否需更新累计量了
    first_epoch 如果是第一个正式训练的epoch，history accu使用历史prediction的平均
    only_infer 如果是True，代表只做inference，用来累计历史
    '''
    global NOTSAME_SELECT_THRESHOLD,SAME_SELECT_THRESHOLD
    log_meta_loss = 0.0
    meta_loss_log = []
    meta_ema_loss_log = []
    meta_cur_loss_log = []
    meta_weight_gt_log = []
    meta_weight_acc_log = []
    meta_weight_major_log = []
    meta_weight_pred_log = []
    meta_false_refurb_tmp_loss = []
    meta_correct_refurb_tmp_loss = []
    meta_noisy_or_false_refurb_tmp_loss = []
    meta_same_and_noisy_tmp_loss = []
    meta_same_and_clean_tmp_loss = []
    #ema loss
    meta_false_refurb_ema_loss = []
    meta_correct_refurb_ema_loss= []
    meta_noisy_or_false_refurb_ema_loss= []
    meta_same_and_noisy_ema_loss= []
    meta_same_and_clean_ema_loss= []
    #hist entropy
    meta_false_refurb_hist_entropy= []
    meta_correct_refurb_hist_entropy= []
    meta_noisy_or_false_refurb_hist_entropy= []
    meta_same_and_noisy_hist_entropy = []
    meta_same_and_clean_hist_entropy = []
    #cur entropy
    meta_false_refurb_cur_entropy = []
    meta_correct_refurb_cur_entropy = []
    meta_noisy_or_false_refurb_cur_entropy = []
    meta_same_and_noisy_cur_entropy = []
    meta_same_and_clean_cur_entropy = []
    for i in range(C['acc_epochs']):
        meta_model.train()
        meta_model_cls_notsame.train()
        meta_model_cls_same.train()
        meta_same_acc_log = []
        meta_notsame_acc_log = []
        meta_same_ce_loss_log = []
        meta_notsame_ce_loss_log = []
        meta_ema_loss_log = []
        meta_cur_loss_log = []
        meta_weight_gt_log = []
        meta_weight_pred_log = []
        notsame_pred_kl = []
        notsame_noisy_kl = []
        notsame_acc_kl = []
        notsame_gt = []
        same_gt = []
        notsame_probs = []
        same_probs = []
        notsame_preds = []
        same_preds = []
        dev_total_crct_refurb = 1
        dev_total_incrct_refurb = 1
        dev_total_noisy = 1
        total_same_num = 1
        total_same_crct_num = 1
        total_notsame_num = 1
        total_notsame_crct_num = 1
        meta_notsame_total_num = 1
        meta_notsame_noisy_crct_num = 1
        meta_notsame_pred_crct_num = 1
        gt_need_lcs_token_nums = []
        gt_noneed_lcs = 0
        gt_need_lcs = 0
        for dev_batch_idx,(src, tgt_input, tgt_output, enc_mask,is_padding,inds,tgt_input_ori,tgt_output_ori,is_padding_ori) in enumerate(clean_dataloader):
            # total_dev_batch_idx += 1
            cur_len = tgt_output.size(0)
            tgt_len = tgt_output_ori.size(0)
            bsz = tgt_output.size(1)
            src = src.to(device=device, non_blocking=True)
            tgt_input = tgt_input.to(device=device, non_blocking=True)
            tgt_output = tgt_output.to(device=device, non_blocking=True)
            tgt_prob = F.one_hot(tgt_output,num_classes=len(vocab_list)).float().to(device)
            enc_mask = enc_mask.to(device=device, non_blocking=True)
            is_padding = is_padding.to(device=device, non_blocking=True)
            is_padding_origin = is_padding_ori.to(device=device,non_blocking=True)
            tgt_output_ori = tgt_output_ori.to(device=device,non_blocking=True)
            with torch.no_grad():
                hist_loss_dev = np.transpose(hist_losses_dev[inds,:cur_len],(1,0))
                padded_gt_tgt_label,is_padding_ori,need,noneed,need_LCS_token_nums = get_padded_gt(tgt_output,tgt_output_ori,is_padding,is_padding_origin)
                gt_need_lcs_token_nums.extend(need_LCS_token_nums)
                gt_noneed_lcs += noneed
                gt_need_lcs += need
                if not only_infer:
                    hist_preds = np.transpose(hist_track_dev[inds,:cur_len,:cur_fetch_ind],(1,0,2))
                    hist_accu_prob_dev = get_hist_accu_probs(tgt_prob,hist_preds,inds,hist_accu_labels_dev,first_epoch=first_epoch)
                    #get most common predicted label
                    # major_vote_label = get_major_vote_label(hist_preds)
                    # major_vote_prob = F.one_hot(torch.from_numpy(major_vote_label).long(),num_classes=len(vocab_list)).float().to(device)
                #current pred
                logit,_,_,_ = model(src, tgt_input, enc_mask)
                log_prob = F.log_softmax(logit, dim=-1)
                noise_prob = F.softmax(logit,dim=-1)
                cur_pred = torch.argmax(noise_prob,dim=-1)
                same_indexes = cur_pred == tgt_output
                _,cost,score,normalized_cost = calc_loss(log_prob,tgt_prob,is_padding,hist_loss_dev)

                noisy_labels_notsame = tgt_output[~same_indexes] 
                clean_labels_notsame = padded_gt_tgt_label[~same_indexes]
                pred_labels_notsame = cur_pred[~same_indexes]
                pads = (~is_padding_ori)[~same_indexes]
                
                noisy_probs_notsame = tgt_prob[~same_indexes] 
                clean_probs_notsame = F.one_hot(clean_labels_notsame,num_classes=len(vocab_list)).float()
                pred_probs_notsame = noise_prob[~same_indexes]
                pads = (~is_padding_ori)[~same_indexes]
                if not only_infer:
                    acc_probs_notsame = hist_accu_prob_dev[~same_indexes]
                    acc_labels_notsame = torch.argmax(hist_accu_prob_dev,dim=-1)[~same_indexes]
                    acc_kl_div_notsame = F.kl_div(torch.log(acc_probs_notsame + NUMERICAL_EPS),clean_probs_notsame,reduction='none').sum(dim=-1) * pads 
                    notsame_acc_kl.append((acc_kl_div_notsame.sum()/ (~pads).sum()).item())

                pred_kl_div_notsame = F.kl_div(torch.log(pred_probs_notsame + NUMERICAL_EPS),clean_probs_notsame,reduction='none').sum(dim=-1) * pads
                noisy_kl_div_notsame = F.kl_div(torch.log(noisy_probs_notsame + NUMERICAL_EPS),clean_probs_notsame,reduction='none').sum(dim=-1) * pads 
                notsame_pred_kl.append((pred_kl_div_notsame.sum() / (~pads).sum()).item())
                notsame_noisy_kl.append((noisy_kl_div_notsame.sum()/ (~pads).sum()).item())
                
                noisy_same_positions_notsame = (noisy_labels_notsame == clean_labels_notsame) * pads
                pred_same_positions_notsame = (pred_labels_notsame == clean_labels_notsame) * pads
                meta_notsame_noisy_crct_num += noisy_same_positions_notsame.sum()
                meta_notsame_pred_crct_num += pred_same_positions_notsame.sum()
                meta_notsame_total_num += (noisy_same_positions_notsame.sum() + pred_same_positions_notsame.sum())
            if not only_infer:
                hist_entropy, cur_entropy = calc_entropy(hist_preds,cur_pred,is_padding,cur_update_ind)
                if not C['self_train']:
                    weight = meta_model(torch.cat((score[:,:,None],normalized_cost[:,:,None],hist_entropy[:,:,None], cur_entropy[:,:,None]),dim=-1).float(),~is_padding[:,:,None]).squeeze(dim=-1)
                    weight = torch.where(weight.double() < NUMERICAL_EPS,0.0,weight.double()).float() #numerical stability?
                    weight = weight.float()
                    weight_gt = weight[:,:,0][:,:,None]
                    weight_pred = weight[:,:,1][:,:,None]
                    # weight_major = weight[:,:,2][:,:,None]
                    weight_acc = weight[:,:,2][:,:,None]
                    if C['only_different']:
                        weight_gt = weight_gt.squeeze().masked_fill(same_indexes,1.)[:,:,None]
                        weight_pred = weight_pred.squeeze().masked_fill(same_indexes,0.)[:,:,None]
                        # weight_major = weight_major.squeeze().masked_fill(same_indexes,0.)[:,:,None]
                        weight_acc = weight_acc.squeeze().masked_fill(same_indexes,0.)[:,:,None]
                    if C['hist_accumulate']:
                        new_pred_prob = weight_gt * tgt_prob + weight_pred * noise_prob  + weight_acc * hist_accu_prob_dev
                    else:
                        new_pred_prob = weight_gt * tgt_prob + weight_pred * noise_prob 
                else:
                    weight_pred = (torch.ones_like(noise_prob).mean(dim=-1).to(device)[:,:,None] * (~is_padding[:,:,None]))
                    weight_pred = weight_pred.squeeze().masked_fill(same_indexes,0)[:,:,None]
                    new_pred_prob = (1 - weight_pred) * tgt_prob + weight_pred * noise_prob
                if C['new_TF'] or C['hidden_feature']:
                    with torch.no_grad(): 
                        new_tgt_input = torch.argmax(new_pred_prob,dim=-1)
                        start_token = torch.zeros(size=(1,bsz),device=device)
                        new_tgt_input = torch.cat((start_token,new_tgt_input),dim=0).long()[:-1,:]
                        logit,_,total_decode_outputs_new,_ = model(src, new_tgt_input, enc_mask)
                        last_layer_outputs_new = total_decode_outputs_new[-1]
                        tf_log_prob = F.log_softmax(logit, dim=-1)
                        _,tf_loss_old,_,_ = calc_loss(tf_log_prob,tgt_prob,is_padding)
                        if C['new_TF']:
                            log_prob = F.log_softmax(logit[:-1,:,:], dim=-1)
                log_indexes = is_padding_ori | same_indexes
                new_tgt_labels = torch.argmax(new_pred_prob,dim=-1)
                padded_gt_meta_train,is_padding_meta_train,need,noneed,need_LCS_token_nums  = get_padded_gt(new_tgt_labels,tgt_output_ori,is_padding,is_padding_origin)
                padded_gt_tgt_prob = F.one_hot(padded_gt_meta_train,num_classes=len(vocab_list)).float().to(device)
                if C['only_different']:
                    train_meta_indexes = is_padding_meta_train | same_indexes
                    _,meta_loss,_,_ = calc_loss(torch.log(new_pred_prob+ NUMERICAL_EPS) ,padded_gt_tgt_prob,train_meta_indexes,None)
                _,tmp_loss,_,_ = calc_loss(log_prob ,new_pred_prob,is_padding,None)
                # log_indexes = is_padding
                if not C['self_train']:
                    meta_weight_gt_log.extend(weight_gt.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    meta_weight_pred_log.extend(weight_pred.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    # meta_weight_major_log.extend(weight_major.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    meta_weight_acc_log.extend(weight_acc.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                meta_ema_loss_log.extend(score[~log_indexes].detach().cpu().numpy().tolist())
                meta_cur_loss_log.extend(normalized_cost[~log_indexes].detach().cpu().numpy().tolist())
                # meta_loss = meta_loss * (~is_padding_ori)
                meta_loss = meta_loss.sum() / (~train_meta_indexes).sum()

                # if meta_loss > 10:
                    # print("in")
                loss_val = meta_loss.item()
                log_meta_loss += loss_val
                meta_loss_log.append(loss_val)
                # total_meta_loss_log.append(loss_val)
                if not C['self_train']:
                    optimizer_meta.zero_grad() #this one causes the bug?????
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 0.25)
                    optimizer_meta.step()
                new_tgt_pred = torch.argmax(new_pred_prob,dim=-1)
                noisy_positions = (tgt_output != padded_gt_tgt_label) * (~is_padding_ori)
                clean_positions = (~noisy_positions) * (~is_padding_ori)
                #compute classification for same and not same
                if C['select']:
                    same_num = (same_indexes & (~is_padding_ori)).sum().item()
                    not_same_num = ((~same_indexes) & (~is_padding_ori)).sum().item()
                    # not_same_clean_labels = clean_labels[~same_indexes]
                    # same_clean_labels = clean_labels[same_indexes]
                    same_and_notpad_indexes = same_indexes & (~is_padding_ori)
                    notsame_and_notpad_indexes = (~same_indexes)&(~is_padding_ori)
                    not_same_clean_labels = clean_positions[notsame_and_notpad_indexes]
                    same_clean_labels = clean_positions[same_and_notpad_indexes]
                    same_gt.extend(same_clean_labels.cpu().tolist())
                    notsame_gt.extend(not_same_clean_labels.cpu().tolist())
                    

                    meta_cls_input_notsame = torch.cat((tf_loss_old.detach()[notsame_and_notpad_indexes][:,None],score.detach()[notsame_and_notpad_indexes][:,None],tmp_loss.detach()[notsame_and_notpad_indexes][:,None],hist_entropy.detach()[notsame_and_notpad_indexes][:,None],normalized_cost.detach()[notsame_and_notpad_indexes][:,None]),dim=-1).float()
                    meta_cls_input_same = torch.cat((tf_loss_old.detach()[same_and_notpad_indexes][:,None],score.detach()[same_and_notpad_indexes][:,None],tmp_loss.detach()[same_and_notpad_indexes][:,None],hist_entropy.detach()[same_and_notpad_indexes][:,None],normalized_cost.detach()[same_and_notpad_indexes][:,None]),dim=-1).float()
                    logits_not_same = meta_model_cls_notsame(meta_cls_input_notsame,softmax=False)
                    logits_same = meta_model_cls_same(meta_cls_input_same,softmax=False)
                    probs_not_same = F.softmax(logits_not_same,dim=-1)
                    probs_same = F.softmax(logits_same,dim=-1)
                    same_probs.extend(probs_same[:,1].cpu().tolist())
                    notsame_probs.extend(probs_not_same[:,1].cpu().tolist())
                    notsame_pred = (probs_not_same[:,1] >= NOTSAME_SELECT_THRESHOLD).int()
                    same_pred = (probs_same[:,1] >= SAME_SELECT_THRESHOLD).int()
                    notsame_preds.extend(notsame_pred.cpu().tolist())
                    same_preds.extend(same_pred.cpu().tolist())
                    total_same_num += same_num
                    total_notsame_num += not_same_num
                    not_same_crct_num = ((notsame_pred == not_same_clean_labels) & (not_same_clean_labels != -1)).sum().item()
                    same_crct_num = ((same_pred == same_clean_labels) & (same_clean_labels != -1)).sum().item()
                    total_same_crct_num += same_crct_num
                    total_notsame_crct_num += not_same_crct_num
                    # meta_same_acc_log.append(same_crct_num / same_num)
                    # meta_notsame_acc_log.append(not_same_crct_num / not_same_num)
                    not_same_ce_loss = ce_loss(logits_not_same,not_same_clean_labels.long())
                    same_ce_loss = ce_loss(logits_same,same_clean_labels.long())
                    meta_same_ce_loss_log.append(same_ce_loss.item())
                    meta_notsame_ce_loss_log.append(not_same_ce_loss.item())

                    optimizer_meta_cls_same.zero_grad()
                    optimizer_meta_cls_notsame.zero_grad()
                    not_same_ce_loss.backward()
                    same_ce_loss.backward()
                    optimizer_meta_cls_same.step()
                    optimizer_meta_cls_notsame.step()
                correct_after_refurb = new_tgt_pred == padded_gt_tgt_label
                false_after_refurb = ~correct_after_refurb
                noisy_num = noisy_positions.sum()
                correct_refurb_positions = correct_after_refurb & noisy_positions
                correct_refurb_num = correct_refurb_positions.sum()
                false_refurb_positions = false_after_refurb & clean_positions
                false_refurb_num = false_refurb_positions.sum()
                dev_total_crct_refurb += correct_refurb_num
                dev_total_incrct_refurb += false_refurb_num
                dev_total_noisy += noisy_num
            #update history 
            if update_history and i == C['acc_epochs'] - 1:
                if not only_infer:
                    update_all_history(inds,cur_len,cur_update_ind,score,cur_pred,hist_accu_labels_dev,new_pred_prob,is_padding,mode="dev")
                else:
                    update_all_history(inds,cur_len,cur_update_ind,score,cur_pred,None,None,is_padding,mode="dev")
            if not only_infer:
                scheduler_meta.step()
                if C['select']:
                    scheduler_meta_cls_notsame.step()
                    scheduler_meta_cls_same.step()
        logger.info("----------------------------------META DEV LOG-----------------------------------------")
        if not only_infer and C['search']:
            notsame_target,notsame_threshold,notsame_precision, notsame_recall, notsame_f1_score,notsame_acc = opt_threshold(notsame_gt,notsame_probs,beta=C['beta'],type=C['search_target'])
            same_target,same_threshold,same_precision, same_recall, same_f1_score,same_acc = opt_threshold(same_gt,same_probs,beta=C['same_beta'],type=C['search_target'])
            NOTSAME_SELECT_THRESHOLD = notsame_threshold
            SAME_SELECT_THRESHOLD = same_threshold
            logger.info(f"SEARCHED notsame F{C['beta']}: {notsame_target *100:.2f}%, ACC: {notsame_acc * 100:.2f}%, PRECISION: {notsame_precision *100:.2f}%, RECALL: {notsame_recall *100:.2f}%, F1: {notsame_f1_score *100:.2f}%")
            logger.info(f"SEARCHED same F{C['beta']}: {same_target *100:.2f}% ,ACC: {same_acc * 100:.2f}%, PRECISION: {same_precision *100:.2f}%, RECALL: {same_recall *100:.2f}%, F1: {same_f1_score *100:.2f}%")
            logger.info(f"NOTSAME_SELECT_THRESHOLD {NOTSAME_SELECT_THRESHOLD}, SAME_SELECT_THRESHOLD {SAME_SELECT_THRESHOLD}")
        log_str = 'Epoch {:d}, rounds {:d}, Meta_Loss {:.4f}'.format(
                    epoch, i, log_meta_loss / (dev_batch_idx+1))
        logger.info(log_str)
        log_meta_loss = 0.0
        
        meta_notsame_ce_loss_log = np.array(meta_notsame_ce_loss_log)
        meta_same_ce_loss_log = np.array(meta_same_ce_loss_log)
        meta_ema_loss_log = np.array(meta_ema_loss_log)
        meta_cur_loss_log = np.array(meta_cur_loss_log)
        meta_weight_gt_log = np.array(meta_weight_gt_log)
        meta_weight_pred_log = np.array(meta_weight_pred_log)
        if not only_infer and C['select'] and not C['search']:
            notsame_precision, notsame_recall, notsame_f1_score,_ = precision_recall_fscore_support(notsame_gt,notsame_preds,average='binary')
            same_precision, same_recall, same_f1_score,_ = precision_recall_fscore_support(same_gt,same_preds,average='binary')
            notsame_acc = accuracy_score(notsame_gt,notsame_preds)
            same_acc = accuracy_score(same_gt,same_preds)
            logger.info(f"notsame F1: {notsame_f1_score *100:.2f}%, ACC: {notsame_acc * 100:.2f}%, PRECISION: {notsame_precision *100:.2f}%, RECALL: {notsame_recall *100:.2f}%")
            logger.info(f"same F1: {same_f1_score *100:.2f}% ,ACC: {same_acc * 100:.2f}%, PRECISION: {same_precision *100:.2f}%, RECALL: {same_recall *100:.2f}% ")
        logger.info(f"meta_ema_loss_mean: {meta_ema_loss_log.mean():.4f},meta_ema_loss_std: {meta_ema_loss_log.std():.4f}")
        logger.info(f"meta_notsame_total_num: {meta_notsame_total_num},meta_notsame_noisy_crct_num: {meta_notsame_noisy_crct_num}, meta_notsame_pred_crct_num: {meta_notsame_pred_crct_num}")
        logger.info(f"noisy crct ratio: {meta_notsame_noisy_crct_num / (meta_notsame_total_num+1)*100:.2f}%, pred crct ratio: {meta_notsame_pred_crct_num / (meta_notsame_total_num+1)*100:.2f}%")
        logger.info(f"noisy KLdiv mean {np.mean(notsame_noisy_kl):.4f}, std {np.std(notsame_noisy_kl):.4f}, pred KLdiv mean {np.mean(notsame_pred_kl):.4f}, std {np.std(notsame_pred_kl):.4f}, acc KLdiv mean {np.mean(notsame_acc_kl):.4f}, std {np.std(notsame_acc_kl):.4f}")
        logger.info(f"not same selection CE Loss {np.mean(meta_notsame_ce_loss_log):.4f}, same selection CE Loss {np.mean(meta_same_ce_loss_log):.4f}")
        logger.info(f"meta_current_loss_mean: {meta_cur_loss_log.mean():.4f},meta_current_loss_std: {meta_cur_loss_log.std():.4f}")
        logger.info(f"meta_weight_gt_mean: {meta_weight_gt_log.mean():.4f},meta_weight_gt_std: {meta_weight_gt_log.std():.4f}")
        logger.info(f"meta_weight_pred_mean: {meta_weight_pred_log.mean():.4f},meta_weight_pred_std: {meta_weight_pred_log.std():.4f}")
        # logger.info(f"meta_weight_major_mean: {meta_weight_major_log.mean():.4f},meta_weight_major_std: {meta_weight_major_log.std():.4f}")
        meta_weight_acc_log = np.array(meta_weight_acc_log)
        logger.info(f"meta_weight_acc_mean: {meta_weight_acc_log.mean():.4f},meta_weight_acc_std: {meta_weight_acc_log.std():.4f}")
        meta_weight_acc_log = []
        logger.info(f"Epoch {epoch}, #meta_total_noisy {dev_total_noisy}, #meta_correct refurb {dev_total_crct_refurb}, #meta_false refurb {dev_total_incrct_refurb},\
         meta_correct_refurb_rate {dev_total_crct_refurb / dev_total_noisy * 100 :.2f}%, meta_incorrect_refurb_ratio {dev_total_incrct_refurb/dev_total_crct_refurb* 100 :.2f}%")
        # meta_weight_major_log = []
        meta_model.eval()
        meta_model_cls_notsame.eval()
        meta_model_cls_same.eval()

max_steps = C['max_step']
for train_round in range(C['multi_rounds']):
    if train_round > 0:
        C['max_step'] = max_steps // 2
        
        optimizer = torch.optim.Adam(model.parameters(), lr=(C['lr']/5))
        optimizer_meta = torch.optim.Adam(meta_model.parameters(), lr=C['meta_lr_cls'])
        optimizer_meta_cls_notsame = torch.optim.Adam(meta_model_cls_notsame.parameters(), lr=C['meta_lr_cls'])
        optimizer_meta_cls_same = torch.optim.Adam(meta_model_cls_same.parameters(), lr=C['meta_lr_cls'])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, C['max_step'], eta_min=C['lr_min'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = C['lr_gamma'])
        scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta, C['max_step']/10, eta_min=C['meta_lr_cls_min'])
        scheduler_meta_cls_notsame = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta_cls_notsame, C['max_step']/10, eta_min=C['meta_lr_cls_min'])
        scheduler_meta_cls_same = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta_cls_same,C['max_step']/10, eta_min=C['meta_lr_cls_min'])
    else:
        C['max_step'] = max_steps
    logger.info(f"new max steps {C['max_step']}")
    train_step = 0
    if train_round == 0:
        last_update_ind = 0 if ((not C['resume_training'] and not C['read_hist_labels']) or train_round >0) else start_epoch - 1
    max_hist_reached = False if train_round == 0 else True
    for epoch in range(start_epoch, 1000):
        if train_step >= C['max_step']:
            logger.info('-' * 100)
            logger.info(f'{train_round} round done')
            break
        if epoch > start_epoch:
            C['acc_epochs'] = 1
        loss_gamma = min(C['loss_gamma_min'] + loss_gamma_step * train_step,C['loss_gamma_max'])
        if epoch >= C['pretrain_epochs']:
            C['epoch_gap'] = 1
        if epoch % C['epoch_gap'] == 0 and epoch != 0:
            last_update_ind += 1
            if last_update_ind >= (max_history_len - 1):
                max_hist_reached = True
                last_update_ind = 0 
        only_infer = True #if True, on devset, we only infer and not update meta models
        if epoch >= C['pretrain_epochs'] or train_round > 0: 
            only_infer = False
        
        cur_update_ind = last_update_ind
        cur_fetch_ind = max_history_len if max_hist_reached else last_update_ind
        
        #update MLP using dev set at the end of each epoch, except the pretrain epochs
        model.train() #use training mode
        
        same_gt = []
        notsame_gt = []
        same_probs = []
        notsame_probs = []
        notsame_preds = []
        same_preds = []
        train_notsame_ce_loss_log = []
        train_same_ce_loss_log = []
        train_notsame_pred_kl = []
        train_notsame_noisy_kl = []
        train_notsame_acc_kl = []
        total_crct_refurb = 1
        total_incrct_refurb = 1
        total_noisy = 1
    

        train_total_notsame_TP=1
        train_total_notsame_FP=1
        train_total_notsame_TN=1
        train_total_notsame_FN=1
        train_total_same_TP =1
        train_total_same_FP =1
        train_total_same_TN =1
        train_total_same_FN =1

        train_notsame_total_num = 1
        train_notsame_noisy_crct_num = 1
        train_notsame_pred_crct_num = 1

        train_total_same_num = 1
        train_total_same_crct_num = 1
        train_total_notsame_num = 1
        train_total_notsame_crct_num = 1

        if epoch < C['pretrain_epochs'] and epoch % C['epoch_gap'] == 0:
            logger.info(f"update history of dev on epoch {epoch}")
            update_meta_models(cur_update_ind,cur_fetch_ind,update_history=True,first_epoch=(epoch==C['pretrain_epochs'] and train_round ==0),only_infer=only_infer)
        elif epoch >= C['pretrain_epochs']: #结束pretrain以后，epochgap变为1
            logger.info(f"update history and meta model of dev on epoch {epoch}")
            update_meta_models(cur_update_ind,cur_fetch_ind,update_history=True,first_epoch=(epoch==C['pretrain_epochs'] and train_round ==0),only_infer=only_infer) #如果第一次开始正式训练，先训练一下meta models
        for batch_idx,( src, tgt_input, tgt_output, enc_mask, is_padding,inds,tgt_input_ori,tgt_output_ori,is_padding_ori) in enumerate(noise_dataloader):
            #after infer stage,every interval, we update the meta models using devset, and not update the history
            # if batch_idx % C['accumulate_interval'] == 0 and not only_infer and C['accumulate_interval'] > 1: 
            #     update_meta_models(cur_update_ind,cur_fetch_ind,update_history=False,first_epoch=(epoch==C['pretrain_epochs'] and train_round ==0),only_infer=only_infer)

            if train_step >= C['max_step']:
                break
            if C['warmup_step'] > 0 and train_step < C['warmup_step'] and train_round == 0:
                optimizer.param_groups[0]['lr'] = C['lr'] * train_step / C['warmup_step']
            src = src.to(device=device, non_blocking=True)
            tgt_input = tgt_input.to(device=device, non_blocking=True)
            tgt_output = tgt_output.to(device=device, non_blocking=True)
            enc_mask = enc_mask.to(device=device, non_blocking=True)
            is_padding = is_padding.to(device=device, non_blocking=True)
            tgt_output_ori = tgt_output_ori.to(device=device, non_blocking=True)
            is_padding_origin = is_padding_ori.to(device=device, non_blocking=True)
            # new_tgt_output = tgt_output
            # noisy_tgt_output = tgt_output
            noisy_tgt_prob = F.one_hot(tgt_output,num_classes=len(vocab_list)).float()
            cur_len,bsz = tgt_output.shape
            tgt_len = tgt_output_ori.size(0)
            padded_gt_tgt_label,is_padding_ori,need,noneed,need_LCS_token_nums  = get_padded_gt(tgt_output,tgt_output_ori,is_padding,is_padding_origin)
            optimizer.zero_grad()
            if C['new_TF'] and not only_infer:
                with torch.no_grad():
                    logit,_,_,_ = model(src, tgt_input, enc_mask)
            else:
                logit,_,_,_ = model(src, tgt_input, enc_mask)
            log_prob = F.log_softmax(logit, dim=-1)
            noise_prob = F.softmax(logit,dim=-1)
            cur_pred = torch.argmax(noise_prob,dim=-1)
            same_indexes = cur_pred == tgt_output
            noisy_labels_notsame = tgt_output[~same_indexes] 
            clean_labels_notsame = padded_gt_tgt_label[~same_indexes]
            pred_labels_notsame = cur_pred[~same_indexes]
            pads = (~is_padding_ori)[~same_indexes]
            noisy_probs_notsame = noisy_tgt_prob[~same_indexes] 
            clean_probs_notsame = F.one_hot(clean_labels_notsame,num_classes=len(vocab_list)).float()
            pred_probs_notsame = noise_prob[~same_indexes]
            
            pred_kl_div_notsame = F.kl_div(torch.log(pred_probs_notsame + NUMERICAL_EPS),clean_probs_notsame,reduction='none').sum(dim=-1) * pads
            noisy_kl_div_notsame = F.kl_div(torch.log(noisy_probs_notsame + NUMERICAL_EPS),clean_probs_notsame,reduction='none').sum(dim=-1) * pads 
            

            train_notsame_pred_kl.append((pred_kl_div_notsame.sum() / (~pads).sum()).item())
            train_notsame_noisy_kl.append((noisy_kl_div_notsame.sum()/ (~pads).sum()).item())

            noisy_same_positions_notsame = (noisy_labels_notsame == clean_labels_notsame) * pads
            pred_same_positions_notsame = (pred_labels_notsame == clean_labels_notsame) * pads
            train_notsame_noisy_crct_num += noisy_same_positions_notsame.sum()
            train_notsame_pred_crct_num += pred_same_positions_notsame.sum()
            train_notsame_total_num += (noisy_same_positions_notsame.sum() + pred_same_positions_notsame.sum())
            hist_loss = np.transpose(hist_losses[inds,:cur_len],(1,0)) #still need hist loss even the first epoch
            #meta model things
            if not only_infer:
                hist_preds = np.transpose(hist_track[inds,:cur_len,:cur_fetch_ind],(1,0,2))
                major_vote_label = get_major_vote_label(hist_preds)
                major_vote_prob = F.one_hot(torch.from_numpy(major_vote_label),num_classes=len(vocab_list)).float().to(device)
                with torch.no_grad():
                    weight,cost,score,normalized_cost = calc_loss(log_prob,noisy_tgt_prob,is_padding,hist_loss)
                    hist_accu_prob = get_hist_accu_probs(noisy_tgt_prob,hist_preds,inds,hist_accu_labels,first_epoch=(epoch==C['pretrain_epochs'] and train_round ==0))
                    acc_labels_notsame = torch.argmax(hist_accu_prob,dim=-1)[~same_indexes]
                    acc_probs_notsame = hist_accu_prob[~same_indexes]
                    acc_kl_div_notsame = F.kl_div(torch.log(acc_probs_notsame + NUMERICAL_EPS),clean_probs_notsame,reduction='none').sum(dim=-1) * pads 
                    train_notsame_acc_kl.append((acc_kl_div_notsame.sum()/ (~pads).sum()).item())
                    hist_entropy, cur_entropy = calc_entropy(hist_preds,cur_pred,is_padding,cur_update_ind)
                    if not C['self_train']:
                        weight = meta_model(torch.cat((score[:,:,None],normalized_cost[:,:,None],hist_entropy[:,:,None], cur_entropy[:,:,None]),dim=-1).float(),~is_padding[:,:,None]).squeeze(dim=-1)
                        weight = weight * (~is_padding[:,:,None])
                        weight = torch.where(weight.double() < NUMERICAL_EPS,0.0,weight.double()).float() #numerical stability?
                        weight_gt = weight[:,:,0][:,:,None]
                        weight_pred = weight[:,:,1][:,:,None]
                        # weight_major = weight[:,:,2][:,:,None]
                        weight_acc = weight[:,:,2][:,:,None]
                        if C['only_different']:
                            weight_gt = weight_gt.squeeze().masked_fill(same_indexes,1.)[:,:,None]
                            weight_pred = weight_pred.squeeze().masked_fill(same_indexes,0.)[:,:,None]
                            # weight_major = weight_major.squeeze().masked_fill(same_indexes,0.)[:,:,None]
                            weight_acc = weight_acc.squeeze().masked_fill(same_indexes,0.)[:,:,None]
                    else:
                        weight_pred = torch.ones_like(noise_prob).mean(dim=-1).to(device)[:,:,None] * (~is_padding[:,:,None])
                        weight_pred = weight_pred.squeeze().masked_fill(same_indexes,0)[:,:,None]
                    log_indexes = is_padding | same_indexes
                    # log_indexes = is_padding
                    weight_gt_log.extend(weight_gt.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    weight_pred_log.extend(weight_pred.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    # weight_major_log.extend(weight_major.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    weight_acc_log.extend(weight_acc.squeeze(dim=-1)[~log_indexes].detach().cpu().numpy().tolist())
                    loss_log.extend(score[~log_indexes].detach().cpu().numpy().tolist())
                    cur_loss_log.extend(normalized_cost[~log_indexes].detach().cpu().numpy().tolist())
                    if C['self_train']:
                        new_tgt_prob = (1 - weight_pred) * noisy_tgt_prob + weight_pred * noise_prob
                    elif C['hist_accumulate']:
                        new_tgt_prob = weight_gt * noisy_tgt_prob + weight_pred * noise_prob + weight_acc * hist_accu_prob
                    else:
                        new_tgt_prob = weight_gt * noisy_tgt_prob + weight_pred * noise_prob   
                if C['hidden_feature']:
                    if C['new_TF']:
                        new_tgt_input = torch.argmax(new_tgt_prob,dim=-1)
                        start_token = torch.zeros(size=(1,bsz),device=device)
                        new_tgt_input = torch.cat((start_token,new_tgt_input),dim=0).long()[:-1,:]
                        logit,_,last_decoder_layer_outputs_new,_ = model(src, new_tgt_input, enc_mask)
                        last_layer_outputs_new = last_decoder_layer_outputs_new[-1]
                        log_prob = F.log_softmax(logit[:-1,:,:], dim=-1)
                    else:
                        with torch.no_grad():
                            new_tgt_input = torch.argmax(new_tgt_prob,dim=-1)
                            start_token = torch.zeros(size=(1,bsz),device=device)
                            new_tgt_input = torch.cat((start_token,new_tgt_input),dim=0).long()[:-1,:]
                            logit,_,last_decoder_layer_outputs_new,_ = model(src, new_tgt_input, enc_mask)
                            last_layer_outputs_new = last_decoder_layer_outputs_new[-1]
                            tf_log_prob = F.log_softmax(logit, dim=-1)
                            _,tf_loss_old,_,_ = calc_loss(tf_log_prob,noisy_tgt_prob,is_padding)
                _,loss,_,_ = calc_loss(log_prob,new_tgt_prob,is_padding)
                if not C['select']:
                    loss = loss.sum() / (~is_padding).sum()
            else:
                weight,loss,score,normalized_cost = calc_loss(log_prob,noisy_tgt_prob,is_padding,hist_loss)
                loss = loss.sum() / (~is_padding).sum()
            if not only_infer:
                new_tgt_pred = torch.argmax(new_tgt_prob,dim=-1)
                noisy_positions = (tgt_output != padded_gt_tgt_label) * (~is_padding_ori)
                clean_positions = (~noisy_positions) * (~is_padding_ori)
                if C['select']:
                    same_num = (same_indexes & (~is_padding_ori)).sum().item()
                    not_same_num = ((~same_indexes) & (~is_padding_ori)).sum().item()
                    # not_same_clean_labels = clean_labels[~same_indexes]
                    # same_clean_labels = clean_labels[same_indexes]
                    same_and_notpad_indexes = same_indexes & (~is_padding_ori)
                    notsame_and_notpad_indexes = (~same_indexes)&(~is_padding_ori)
                    not_same_clean_labels = clean_positions[notsame_and_notpad_indexes]
                    same_clean_labels = clean_positions[same_and_notpad_indexes]
                    same_gt.extend(same_clean_labels.cpu().tolist())
                    notsame_gt.extend(not_same_clean_labels.cpu().tolist())
                    
                    meta_cls_input_notsame = torch.cat((tf_loss_old.detach()[notsame_and_notpad_indexes][:,None],score.detach()[notsame_and_notpad_indexes][:,None],loss.detach()[notsame_and_notpad_indexes][:,None],hist_entropy.detach()[notsame_and_notpad_indexes][:,None],normalized_cost.detach()[notsame_and_notpad_indexes][:,None]),dim=-1).float()
                    meta_cls_input_same = torch.cat((tf_loss_old.detach()[same_and_notpad_indexes][:,None],score.detach()[same_and_notpad_indexes][:,None],loss.detach()[same_and_notpad_indexes][:,None],hist_entropy.detach()[same_and_notpad_indexes][:,None],normalized_cost.detach()[same_and_notpad_indexes][:,None]),dim=-1).float()
                    logits_not_same = meta_model_cls_notsame(meta_cls_input_notsame,softmax=False)
                    logits_same = meta_model_cls_same(meta_cls_input_same,softmax=False)
                    probs_not_same = F.softmax(logits_not_same,dim=-1)
                    probs_same = F.softmax(logits_same,dim=-1)
                    same_probs.extend(probs_same[:,1].cpu().tolist())
                    notsame_probs.extend(probs_not_same[:,1].cpu().tolist())
                    notsame_pred = (probs_not_same[:,1] >= NOTSAME_SELECT_THRESHOLD).int()
                    same_pred = (probs_same[:,1] >= SAME_SELECT_THRESHOLD).int()
                    notsame_preds.extend(notsame_pred.cpu().tolist())
                    same_preds.extend(same_pred.cpu().tolist())
                    
                    not_same_ce_loss = ce_loss(logits_not_same,not_same_clean_labels.long())
                    same_ce_loss = ce_loss(logits_same,same_clean_labels.long())
                    train_same_ce_loss_log.append(same_ce_loss.item())
                    train_notsame_ce_loss_log.append(not_same_ce_loss.item())

                    loss_same_part = loss[same_and_notpad_indexes] * same_pred.bool()
                    loss_same_part_avg = loss_same_part.sum() / same_pred.bool().sum() 
                    loss_not_same_part = loss[notsame_and_notpad_indexes] * notsame_pred.bool()
                    loss_not_same_part_avg = loss_not_same_part.sum() / notsame_pred.bool().sum()
                    loss = loss_same_part_avg + loss_not_same_part_avg


            log_loss += loss.item()
            loss.backward()
            optimizer_meta_cls_notsame.zero_grad()
            optimizer_meta_cls_same.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            if train_step >= C['warmup_step']:
                scheduler.step()
            train_step += 1
            noisy_positions = (tgt_output != padded_gt_tgt_label) * (~is_padding_ori)
            clean_positions = (~noisy_positions) * (~is_padding_ori)
            if not only_infer:
                #refurb metrics
                correct_after_refurb = new_tgt_pred == padded_gt_tgt_label
                false_after_refurb = ~correct_after_refurb
                noisy_num = noisy_positions.sum()
                correct_refurb_num = (correct_after_refurb & noisy_positions).sum()
                false_refurb_num = (false_after_refurb & clean_positions).sum()
                total_crct_refurb += correct_refurb_num
                total_incrct_refurb += false_refurb_num
                total_noisy += noisy_num
            #update history 
            if epoch % C['epoch_gap'] == 0:
                if not only_infer:
                    update_all_history(inds,cur_len,cur_update_ind,score,cur_pred,hist_accu_labels,new_tgt_prob,is_padding,mode="train")
                else:
                    update_all_history(inds,cur_len,cur_update_ind,score,cur_pred,None,None,is_padding,mode="train")
        #after train epoch. update dev epoch history
        # update_meta_models(cur_update_ind,cur_fetch_ind,update_history=True,first_epoch=(epoch==C['pretrain_epochs'] and train_round ==0),only_infer=only_infer)
        
        #log common info
        elapsed = time.time() - log_start_time
        logger.info("-------------------------------------TRAINING LOG------------------------------------")
        log_str = 'Round {:d},Epoch {:d}, Step {:d}, Batch {:d}, loss_gamma {:3f} | Speed {:.2f}ms/it, LR {:3g} | metaLR {:5f} | Loss {:.4f},last_update_ind {:d},cur_fetch_ind {:d},cur_update_ind {:d},epoch_gap {:d}'.format(
                    train_round, epoch, train_step, batch_idx+1, loss_gamma,
                    elapsed * 1000 / (batch_idx+1),
                    optimizer.param_groups[0]['lr'],
                    optimizer_meta.param_groups[0]['lr'],
                    log_loss / (batch_idx+1),
                    last_update_ind,cur_fetch_ind,cur_update_ind,C['epoch_gap'])
        logger.info(log_str)
        log_loss  = 0.0
        log_start_time = time.time()
        loss_log = np.array(loss_log)
        cur_loss_log = np.array(cur_loss_log)
        weight_gt_log = np.array(weight_gt_log)
        weight_pred_log = np.array(weight_pred_log)
        # weight_major_log = np.array(weight_major_log)
        if not only_infer and C['select']:
            notsame_precision, notsame_recall, notsame_f1_score,_ = precision_recall_fscore_support(notsame_gt,notsame_preds,average='binary')
            same_precision, same_recall, same_f1_score,_ = precision_recall_fscore_support(same_gt,same_preds,average='binary')
            if C['search'] and C['beta'] != 1:
                notsame_fbeta_score = ((1+C['beta']) * notsame_precision * notsame_recall)  / (C['beta'] * notsame_precision + notsame_recall)
                same_fbeta_score = ((1+C['beta']) * same_precision * same_recall)  / (C['beta'] * same_precision + same_recall)
                logger.info(f"notsame F{C['beta']} score: {notsame_fbeta_score *100:.2f}%, NOTSAME_SELECT_THRESHOLD {NOTSAME_SELECT_THRESHOLD}, same F{C['beta']} score: {same_fbeta_score *100:.2f}%, SAME_SELECT_THRESHOLD {SAME_SELECT_THRESHOLD}")
            notsame_acc = accuracy_score(notsame_gt,notsame_preds)
            same_acc = accuracy_score(same_gt,same_preds)
            logger.info(f"notsame F1: {notsame_f1_score *100:.2f}%, ACC: {notsame_acc * 100:.2f}%, PRECISION: {notsame_precision *100:.2f}%, RECALL: {notsame_recall *100:.2f}%")
            logger.info(f"same F1: {same_f1_score *100:.2f}% ,ACC: {same_acc * 100:.2f}%, PRECISION: {same_precision *100:.2f}%, RECALL: {same_recall *100:.2f}% ")
        logger.info(f"ema_loss_mean: {loss_log.mean():.4f},ema_loss_std: {loss_log.std():.4f}")
        logger.info(f"train_notsame_total_num: {train_total_notsame_num},train_notsame_noisy_crct_num: {train_notsame_noisy_crct_num}, train_notsame_pred_crct_num: {train_notsame_pred_crct_num}")
        logger.info(f"noisy crct ratio: {train_notsame_noisy_crct_num / (train_notsame_total_num+1)*100:.2f}%, pred crct ratio: {train_notsame_pred_crct_num / (train_notsame_total_num+1)*100:.2f}%")
        logger.info(f"noisy KLdiv mean {np.mean(train_notsame_noisy_kl):.4f}, std {np.std(train_notsame_noisy_kl):.4f}, pred KLdiv mean {np.mean(train_notsame_pred_kl):.4f}, std {np.std(train_notsame_pred_kl):.4f}, acc KLdiv mean {np.mean(train_notsame_acc_kl):.4f}, std {np.std(train_notsame_acc_kl):.4f}")
        logger.info(f"not same selection CE Loss {np.mean(train_notsame_ce_loss_log):.4f}, same selection CE Loss {np.mean(train_same_ce_loss_log):.4f}")
        logger.info(f"current_loss_mean: {cur_loss_log.mean():.4f},current_loss_std: {cur_loss_log.std():.4f}")
        logger.info(f"weight_gt_mean: {weight_gt_log.mean():.4f},weight_gt_std: {weight_gt_log.std():.4f}")
        logger.info(f"weight_pred_mean: {weight_pred_log.mean():.4f},weight_pred_std: {weight_pred_log.std():.4f}")
        # logger.info(f"weight_major_mean: {weight_major_log.mean():.4f},weight_major_std: {weight_major_log.std():.4f}")
        logger.info(f"weight_acc_mean: {np.mean(weight_acc_log):.4f},weight_acc_std: {np.std(weight_acc_log):.4f}")
        train_notsame_ce_loss_log = []
        train_same_ce_loss_log = []
        train_notsame_pred_kl = []
        train_notsame_noisy_kl = []
        train_notsame_acc_kl = []
        train_notsame_total_num = 1
        train_notsame_noisy_crct_num = 1
        train_notsame_pred_crct_num = 1
        train_total_same_num = 1
        train_total_same_crct_num = 1
        train_total_notsame_num = 1
        train_total_notsame_crct_num = 1
        loss_log = []
        cur_loss_log = []
        weight_gt_log = []
        weight_pred_log = []
        # weight_major_log = []
        weight_acc_log = []
        #evaluate model
        # if i == C['acc_epochs'] - 1:
        logger.info(f"Epoch {epoch}, #total_noisy {total_noisy}, #correct refurb {total_crct_refurb}, #false refurb {total_incrct_refurb},\
         correct_refurb_rate {total_crct_refurb / total_noisy* 100 :.2f}%, incorrect_refurb_ratio {total_incrct_refurb/total_crct_refurb* 100 :.2f}%")
        if not C['debug']:
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
                    train_step // C['eval_interval'], train_step, time.time() - eval_start_time, cer)
                logger.info(log_str)
                if best_val_score is None or cer <= best_val_score:
                    torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pt'))
                    torch.save(cer, os.path.join(work_dir, 'best_cer.pt'))
                    best_val_score = cer
                    logger.info('Better CER detected, model saving...')
                    torch.save(optimizer.state_dict(), os.path.join(work_dir, 'best_opt.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(work_dir, 'best_sch.pt'))
                    torch.save(train_step, os.path.join(work_dir, 'best_step.pt'))
                    if C['save_hist']:
                        np.save(os.path.join(work_dir,"hist_track_dev"),hist_track_dev)
                        np.save(os.path.join(work_dir,"hist_losses_dev"),hist_losses_dev)
                        np.save(os.path.join(work_dir,"hist_track"),hist_track)
                        np.save(os.path.join(work_dir,"hist_losses"),hist_losses)
                        if C['hist_accumulate']:
                            np.save(os.path.join(work_dir,"hist_accu_labels"),hist_accu_labels)
                            np.save(os.path.join(work_dir,"hist_accu_labels_dev"),hist_accu_labels_dev)
                    torch.save(meta_model.state_dict(),os.path.join(work_dir, 'best_meta_model.pt'))
                    torch.save(optimizer_meta.state_dict(),os.path.join(work_dir, 'best_meta_optimizer.pt'))
                    torch.save(scheduler_meta.state_dict(),os.path.join(work_dir, 'best_meta_scheduler.pt'))

                    torch.save(meta_model_cls_notsame.state_dict(),os.path.join(work_dir, 'best_meta_model_cls_notsame.pt'))
                    torch.save(optimizer_meta_cls_notsame.state_dict(),os.path.join(work_dir, 'best_meta_optimizer_meta_cls_notsame.pt'))
                    torch.save(scheduler_meta_cls_notsame.state_dict(),os.path.join(work_dir, 'best_meta_scheduler_meta_cls_notsame.pt'))

                    torch.save(meta_model_cls_same.state_dict(),os.path.join(work_dir, 'best_meta_model_cls_same.pt'))
                    torch.save(optimizer_meta_cls_same.state_dict(),os.path.join(work_dir, 'best_meta_optimizer_meta_cls_same.pt'))
                    torch.save(scheduler_meta_cls_same.state_dict(),os.path.join(work_dir, 'best_meta_scheduler_meta_cls_same.pt'))
                    torch.save(model.state_dict(), os.path.join(work_dir, 'model-{}.state.pt'.format(cur_save_id)))
                    cur_save_id = (cur_save_id + 1) % C['num_saved_models']
                    logger.info(f"better model detected, model saved....,save as {cur_save_id} state")
            model.train()
        # draw_meta_stats(None,None,total_dev_batch_idx,work_dir,total_meta_loss_log,epoch,None,"total")

##########
# test
##########

load_path =  os.path.join(work_dir, 'best_model.pt')
model.load_state_dict(torch.load(load_path))
model = model.to(device)
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