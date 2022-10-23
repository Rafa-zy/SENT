import os
import json
import codecs
import random
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support,accuracy_score,precision_recall_curve

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def plot_grad_flow(named_parameters, path):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if(p is not None):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean() if p.grad is not None else 0)
                max_grads.append(p.grad.abs().max() if p.grad is not None else 0)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    #plt.tight_layout()
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # plt.show()
    plt.savefig(path)
    return plt, max_grads


def read_manifest(mode, manifest_path, max_duration=float('inf'), min_duration=0.2, max_text_len=float('inf')):
    manifest = []
    durations = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for json_line in f:
            json_data = json.loads(json_line)
            if (json_data["duration"] <= max_duration and
                    json_data["duration"] >= min_duration and
                    len(json_data["text"]) <= max_text_len):
                manifest.append(json_data)
                durations.append(json_data['duration'])
    if mode == 'train' or mode == 'dev':
        data = list(zip(manifest, durations))
        data = sorted(data, key=lambda x:x[1])
        manifest, durations = zip(*data)
    return manifest, durations

def read_manifest_(manifest_path, max_duration=float('inf'), min_duration=0.2, max_text_len=float('inf')):
    manifest = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for json_line in f:
            json_data = json.loads(json_line)
            if (json_data["duration"] <= max_duration and
                    json_data["duration"] >= min_duration and
                    len(json_data["text"]) <= max_text_len):
                manifest.append(json_data)
    return manifest

def load_vocabulary(vocab_filepath):
    vocab_lines = []
    with open(vocab_filepath, 'r', encoding='utf-8') as file:
        vocab_lines.extend(file.readlines())
    vocab_list = ['<s>'] + [line[:-1] for line in vocab_lines]
    vocab_dict = dict([(token, id) for (id, token) in enumerate(vocab_list)])
    return vocab_dict, vocab_list

def load_mt_vocabulary(vocab_filepath):
    vocab_lines = []
    with open(vocab_filepath, 'r', encoding='utf-8') as file:
        vocab_lines.extend(file.readlines())
    vocab_list = [line.strip() for line in vocab_lines]
    vocab_dict = dict([(token, id) for (id, token) in enumerate(vocab_list)])
    return vocab_dict, vocab_list


def read_log(log_dir):
    res_dict = {
        "| Loss":[],
        "rbf_lr_center":[],
        "rbf_lr_sigma":[],
        "mu_mean":[],
        "sigma_mean":[],
        "Epoch":[],
        "Step":[]
    }
    with open(log_dir,'r',encoding="utf-8") as f:
        check_list = list(res_dict.keys())
        str_len = [8,8,6,6,5,10]
        skip_config = 0
        for index,line in enumerate(f.readlines()):
            if "======" in line:
                skip_config += 1
                continue
            if skip_config < 2:
                continue
            for i,check_comp in enumerate(check_list):
                if check_comp in line and "Eval" not in line and "Better" not in line:
                    start_index = line.find(check_comp) + len(check_comp)
                    cut_str = line[start_index:(start_index+str_len[i])]
                    if ":" in cut_str:
                        info = cut_str.split(":")
                        res = float(info[1])
                    if "," in cut_str:
                        info = cut_str.split(",")
                        res = int(info[0])
                    res_dict[check_comp].append(res)
            # if len(res_dict["Step"]) > len(res_dict["mu"]):
                # print("in")

    return res_dict

def draw_mu_sigma(mu,sigma,steps):
    x = steps
    y1 = mu
    y2 = sigma

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x, y1, label='mu')
    ax1.set_ylabel('mu')
    ax1.set_title("mu & sigma")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'r', label='sigma')
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel('sigma')
    ax1.legend()
    ax2.legend()
    plt.savefig("test_mu_sigma.png")
    # plt.show()


def read_step_meta(file_dir,suffix="loss"):
    if suffix == "mu":
        fl_dir = file_dir + "step_mu.txt"
    elif suffix == "sigma":
        fl_dir = file_dir + "step_sigma.txt"
    elif suffix == "loss": 
        fl_dir = file_dir + "step_metaloss.txt"
    else:
        raise ValueError("no such suffix")
    with open(fl_dir,'r') as f:
        info = f.readline()
        tmp = info.split(",")
        data = np.array(tmp)
        length = len(data)
        gap = 500
        groups = length // gap
        final_index = groups * gap
        data = data[:final_index]
        data = np.split(data,groups)
        res = []
        for i,d in enumerate(data):
            # if i >= 100:
            #     break
            x = np.arange(gap)
            try:
                y = d.astype(np.float)
            except:
                continue
            # if np.isnan(y.mean()) or i <= 30:
                # continue
            res.append(y.mean())    

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(x, y, label='meta')
            ax1.set_ylabel(f"meta_{suffix}")
            ax1.set_title(f"meta_{suffix}")
            plt.savefig(f"{file_dir}meta_{suffix}_{i}.png")

        x = np.arange(len(res))
        y = res
        # plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y, label=f'meta_{suffix}')
        ax1.set_ylabel(f'meta_{suffix}')
        ax1.set_title(f"meta_{suffix}")
        plt.savefig(f"{file_dir}meta_{suffix}_mean.png")



def min_max_normalize(target,is_padding):
    target = target.masked_fill(is_padding,torch.finfo(target.dtype).max)
    min_val = torch.min(target)
    target = target.masked_fill(is_padding,0.)
    max_val = torch.max(target)
    normalized_target = (target - min_val.detach()) / ((max_val - min_val) + torch.finfo(target.dtype).eps).detach()
    return normalized_target * (~is_padding)



def draw_meta_stats(meta_mu,meta_sigma,steps,work_dir,meta_loss,epoch,batch_idx,accstep,prefix="metaloss"):
    # x = np.arange(steps)
    x = np.arange(len(meta_loss))
    try:
        meta_loss_array = np.array(meta_loss)
        filtered_indexes = is_outlier(meta_loss_array)
        filtered_loss = meta_loss_array[~filtered_indexes]
        filtered_x = x[~filtered_indexes]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax1.plot(x, meta_loss, label='meta_loss')
        ax1.plot(filtered_x,filtered_loss,label='meta_loss')
        ax1.set_ylabel(f"meta_loss")
        ax1.set_title(f"meta_loss")
        plt.savefig(os.path.join(work_dir,f"{prefix}_epoch{epoch}_{accstep}accstep.png"))
        plt.close(fig)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.plot(x, meta_mu, label='meta_mu')
        # ax1.set_ylabel(f"meta_mu")
        # ax1.set_title(f"meta_mu")
        # plt.savefig(f"{work_dir}meta_mu_{epoch}_{batch_idx}.png")
        # plt.close(fig)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.plot(x, meta_sigma, label='meta_sigma')
        # ax1.set_ylabel(f"meta_sigma")
        # ax1.set_title(f"meta_sigma")
        # plt.savefig(f"{work_dir}meta_sigma_{epoch}_{batch_idx}.png")
        # plt.close(fig)
    except Exception:
        return




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


def get_history_labels(inds,hist_labels):
    prob_list = []
    for idx,index in enumerate(inds):
        prob_list.append(torch.from_numpy(hist_labels[index]))
    return pad_sequence(prob_list,batch_first=False)

def update_history_labels(inds,hist_labels,new_prob,mask,log=False,**kwargs):
    for idx,index in enumerate(inds):
        actual_len = (~(mask[:,idx])).sum(dim=0)
        prob_tmp = new_prob[:actual_len,idx]
        hist_labels[index] = prob_tmp.detach().cpu().clone().numpy().astype(np.float32)
    return True

def compute_F1(TP,TN,FP,FN):
    all_neg_pred = TN + FN
    all_pos_pred = TP + FP
    all_pos_true = TP + FN
    all_crct = TP + TN
    acc = all_crct / (all_neg_pred + all_pos_pred)
    precision = TP / all_pos_pred
    recall = TP / all_pos_true
    F1 = (2 * precision * recall) / (precision + recall)
    return acc, precision, recall, F1

class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, mask):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0
        # True for non-padded position, False for padded positions
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        n = mask.sum()
        mask = mask / n

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp

import numpy as np 
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support,accuracy_score

def opt_threshold(y_true, y_probs,beta=1,type='f'):
    y_true = np.array(y_true)
    y_scores = np.array(y_probs)
    if type == 'auc':
        auc = roc_auc_score(y_true, y_scores)
        fpr,tpr,thresholds = roc_curve(y_true,y_scores)
        threshold = thresholds[np.argmax(tpr-fpr)]
        target = auc
    else:
        precision, recall, thresholds = precision_recall_curve(y_true,y_scores)
        f_beta_scores = ((1+beta) * precision * recall)  / (beta * precision + recall)
        target_idx = np.argmax(f_beta_scores)
        threshold_idx = max(0,target_idx-1)
        threshold,target = thresholds[threshold_idx],f_beta_scores[target_idx]
    y_pred = np.zeros_like(y_true)        
    sift_pos = y_scores >= threshold
    y_pred[sift_pos] = 1
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    if beta != 1:
        f_score = ((1+beta) * precision * recall)  / (beta * precision + recall)
    acc = accuracy_score(y_true,y_pred)
    return target,threshold,precision, recall, f_score,acc

# if __name__ == "__main__":
#     y_true = [1,   0,   1,   0,  1,  1,0,0,1,1,0]
#     y_pred = [0.62,0.3,0.8,0.6,0.7,0.9,0.3,0.5,0.64,0.1,0.8]
#     target,threshold,precision, recall, f1_score,acc = opt_threshold(y_true,y_pred,'f1')
#     print(1)

# if __name__ == "__main__":
    # read_step_meta("/mfs/wangzhihao/research/noisy_label_higher/exp_debug_cuishou_1e-1lr_500acc_nonfix/","mu")
    # read_step_meta("/mfs/wangzhihao/research/noisy_label_higher/exp_debug_cuishou_1e-3lr_500acc_nonfix/","loss")
    # read_step_meta("/mfs/wangzhihao/research/noisy_label_higher/exp_debug_cuishou_1e-1lr_500acc_nonfix/","sigma")
    # res_dict = read_log("/mfs/wangzhihao/research/noisy_label_higher/exp_debug_bigacc_nonfix_pre/log.txt")
    # draw_mu_sigma(res_dict["mu_mean"],res_dict["sigma_mean"],res_dict["Step"])
