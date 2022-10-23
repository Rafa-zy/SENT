from __future__ import print_function, division
import torch
from torch.nn.functional import threshold
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from util.utils import *
import numpy as np

class Evaluator(object):
    """ Class to evaluate models with given datasets.
    """
    def __init__(self, loss,logger, config):
        self.loss = loss
        self.logger= logger
        self.config = config
        self.batch_size = self.config.valid_batch_size
        self.device = self.config.device
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    def infer(self,model,data_loader):
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                targets = batch['labels'].to(self.device, dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    outputs = model(ids, attention_mask, token_type_ids, labels=targets)
                else:
                    outputs = model(ids, attention_mask, labels=targets)
                loss, logits = outputs[0], outputs[1]
    
    def evaluate(self, model, data_loader, is_test=False, return_details=False):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model: model to evaluate
            data: dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        loss = self.loss
        model.eval()
        
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0
        n_correct, n_total = 0, 0
        precision_ls, recall_ls, f1_ls, support_ls = [], [], [], []
        pred_ls, target_ls = np.array([]), np.array([])
        ids_ls = np.array([])
        with torch.no_grad():
            for _, (src,targets,inds,ori_targets) in enumerate(data_loader):
                ids = torch.tensor(src['input_ids']).to(self.device, dtype=torch.long)
                attention_mask = torch.tensor(src['attention_mask']).to(self.device, dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    token_type_ids = torch.tensor(src['token_type_ids']).to(self.device, dtype=torch.long)
                targets = ori_targets.to(self.device, dtype=torch.long)

                if self.config.pretrained_model == "bert":
                    outputs = model(ids, attention_mask, token_type_ids, labels=targets)
                else:
                    outputs = model(ids, attention_mask, labels=targets)
                loss, logits = outputs[0], outputs[1]
                if is_test:
                    eval_loss += 0
                else:
                    eval_loss += loss.item()
                big_val, big_idx = torch.max(logits.data, dim=-1) # [bsz,1]
                n_correct += self.calculate_accu(big_idx, targets)
                
                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
                target_ls = np.append(target_ls, targets.cpu().numpy())
                pred_ls = np.append(pred_ls, big_idx.cpu().numpy())
                ids_ls = np.append(ids_ls, inds.cpu().numpy())
                # if _ % 1000 == 0:
                #     loss_step = eval_loss / nb_eval_steps
                #     accu_step = (n_correct)/nb_eval_examples
                #     if is_test == True:
                #         self.logger.info(f"Test Loss per 1000 steps: {loss_step}")
                #         self.logger.info(f"Test Accuracy per 1000 steps: {accu_step}")
                #     else:
                #         self.logger.info(f"Validation Loss per 1000 steps: {loss_step}")
                #         self.logger.info(f"Validation Accuracy per 1000 steps: {accu_step}")
            
        epoch_loss = eval_loss / nb_eval_steps
        epoch_accu = (n_correct) / nb_eval_examples
        target_arr, pred_arr, ids_arr = target_ls.flatten(), pred_ls.flatten(), ids_ls.flatten()
        # print(target_arr.shape)
        # print(target_arr)
        # print(pred_arr.shape)
        precision, recall, f1_score, support = precision_recall_fscore_support(target_arr, pred_arr, average='weighted')

        if is_test == True:
            # self.logger.info(f"Test Loss Epoch: {epoch_loss}")
            self.logger.info(f"Test Accuracy Epoch: {epoch_accu}")
            self.logger.info(f"Test F1-score Epoch: {f1_score}")
            self.logger.info(f"Test Precision Epoch: {precision}")
            self.logger.info(f"Test Recall Epoch: {recall}")
        else:
            self.logger.info(f"Validation Loss Epoch: {epoch_loss}")
            self.logger.info(f"Validation Accuracy Epoch: {epoch_accu}")
            self.logger.info(f"Validation F1-score Epoch: {f1_score}")
            self.logger.info(f"Validation Precision Epoch: {precision}")
            self.logger.info(f"Validation Recall Epoch: {recall}")
        if return_details == True:
            return epoch_loss, epoch_accu, list(zip(pred_arr, target_arr, ids_arr))
        else:
            return epoch_loss, epoch_accu

    def evaluate_meta(self, model, data_loader,hist_losses,hist_track,hist_prob,instant_losses,instant_simi, calc_entropy_func,loss_func,is_test=False):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model: model to evaluate
            data: dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        loss = self.loss
        model.eval()
        
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0
        n_correct, n_total = 0, 0
        
        with torch.no_grad():
            if self.config.use_entropy:
                hist_entropy,cur_entropy = calc_entropy_func(hist_track)
                hist_entropy,cur_entropy = hist_entropy.to(self.device),cur_entropy.to(self.device)
            if self.config.soft_entropy:
                if self.config.soft_entropy_mul:
                    hist_entropy_soft, hist_entropy_soft_mul = calc_entropy_func(hist_prob,type='soft')
                    hist_entropy_soft_mul[torch.isinf(hist_entropy_soft_mul)] = 0
                    hist_entropy_soft_mul[torch.isnan(hist_entropy_soft_mul)] = 0
                    hist_entropy_soft_mul = hist_entropy_soft_mul.to(self.device)
                else:
                    hist_entropy_soft, _ = calc_entropy_func(hist_prob,type='soft')
                hist_entropy_soft[torch.isinf(hist_entropy_soft)] = 0
                hist_entropy_soft[torch.isnan(hist_entropy_soft)] = 0
                hist_entropy_soft = hist_entropy_soft.to(self.device)
            hist_loss_all,cur_loss_all = torch.from_numpy(hist_losses[:,-1]).to(self.device),torch.from_numpy(instant_losses[:,-1]).to(self.device)
            if self.config.use_simi:
                cur_simi_all = torch.from_numpy(instant_simi[:,-1]).to(self.device)
            meta_preds = []
            meta_gt = []
            meta_probs = []
            for _, (src,targets,inds,ori_targets) in enumerate(data_loader):
                hist_loss,cur_loss = hist_loss_all[inds],cur_loss_all[inds]
                targets = targets.to(self.device,dtype=torch.long)
                ori_targets = ori_targets.to(self.device, dtype=torch.long)
                eval_targets = targets == ori_targets 
                input_signals = []
                if self.config.use_loss:
                    if self.config.use_ema_loss:
                        input_signals = [hist_loss[:,None] if len(hist_loss.shape) == 1 else hist_loss,cur_loss[:,None] if len(cur_loss.shape) == 1 else cur_loss]
                    else:
                        input_signals = [cur_loss[:,None] if len(cur_loss.shape) == 1 else cur_loss]
                if self.config.use_entropy:
                    if self.config.use_hist_entropy:
                        input_signals.extend([hist_entropy[inds][:,None],cur_entropy[inds][:,None]])
                    else:
                        input_signals.extend([cur_entropy[inds][:,None]])
                if self.config.use_simi:
                    cur_simi = cur_simi_all[inds]
                    input_signals.append(cur_simi[:,None] if len(cur_simi.shape) == 1 else cur_simi)
                if self.config.soft_entropy:
                    input_signals.extend([hist_entropy_soft[inds][:,None]])
                if self.config.soft_entropy_mul:
                    for i in range(self.config.class_num):
                        hist_entropy_soft_mul_i = hist_entropy_soft_mul[:,i]
                        input_signals.extend([hist_entropy_soft_mul_i[inds][:,None]])
                if self.config.test_single_feat:
                    feat_name_dict = {"hist_emaloss":0,"cur_loss":1,"hist_entropy":2,"cur_entropy":3,"soft_entropy":4,"cur_simi":5}
                    input_feature = input_signals[feat_name_dict[self.config.test_single_feat_name]]
                    meta_logits = model(input_feature.float())
                else:
                    meta_logits = model(torch.cat(input_signals,dim=-1).float())
                meta_prob = F.softmax(meta_logits,dim=-1)
                meta_probs.extend(meta_prob[:,1].cpu().cpu().tolist())
                loss = loss_func(meta_logits,eval_targets.long())
                big_idx = torch.argmax(meta_logits, dim=-1)
                meta_preds.extend(big_idx.long().cpu().tolist())
                meta_gt.extend(eval_targets.long().cpu().tolist())

                eval_loss += loss.item()
                n_correct += self.calculate_accu(big_idx, eval_targets)
                
                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
        optimize_target,threshold,precision, recall, f1_score,epoch_accu,_,_ = opt_threshold(meta_gt,meta_probs,self.config.meta_target)
        # precision, recall, f1_score, support = precision_recall_fscore_support(meta_gt, meta_preds, average='binary')
        epoch_loss = eval_loss / nb_eval_steps

        return epoch_loss, epoch_accu,precision, recall, f1_score,optimize_target,threshold