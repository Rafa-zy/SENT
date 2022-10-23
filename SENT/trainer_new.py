import os
import torch
from evaluator_new import Evaluator
from torch.utils.data import  DataLoader
from torch.nn import CrossEntropyLoss, MSELoss,DataParallel
from util.early_stopping import EarlyStopping
from util.lexicon_util import stop_words
from transformers import BertTokenizer, BertForSequenceClassification
# from weak_supervision import guide_pseudo_labeling
from nltk.stem import WordNetLemmatizer
from model import BERT_ATTN,MLP_head
from util.datasetxl import *
from util.utils import *
from collections import Counter
import torch.nn.functional as F
from entropy_estimators import continuous
from torchmeta.utils.gradient_based import gradient_update_parameters
import higher
import time
# from util.augment import *
import random
import copy
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,roc_curve

NUMERICAL_EPS = 1e-12

class Trainer(object):
    def __init__(self, config, model,logger, criterion, optimizer, 
                 save_path, train_dataset,dev_dataset,meta_dataset,total_dev_dataset,unlabel_dataset, test_dataset,label_mapping,model_type, do_augment):
        self.config = config
        self.logger = logger
        self.loss = criterion
        self.trec = False
        if self.trec:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.1,0.1,0.2,0.15,0.15])).to(self.device)
        self.evaluator = Evaluator(loss=self.loss,logger=self.logger,config=self.config)
        self.optimizer = optimizer
        self.device = self.config.device
        self.device_meta = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.global_best_model_ls = []
        self.global_best_model = None
        self.best_meta_model = None
        self.best_meta_thresh = None
        self.early_stop_model = None
        self.early_stop_epoch = self.config.hist_len
        self.model_type = model_type
        self.do_augment = do_augment
        
        #+1 for using similarity
        if self.config.use_loss:
            self.config.input_size += 1
            if self.config.use_ema_loss:
                self.config.input_size += 1
        if self.config.use_simi:
            self.config.input_size += 1
        if self.config.use_entropy:
            self.config.input_size += 1
            if self.config.use_hist_entropy:
                self.config.input_size += 1
        if self.config.soft_entropy:
            if self.config.soft_entropy_mul:
                self.config.input_size += self.config.class_num
            self.config.input_size += 1
        if self.config.test_single_feat == True:
            self.config.input_size = 1
        

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.lemmatizer = WordNetLemmatizer()
        self.train_dataset = train_dataset
        self.label_dataset = deepcopy(train_dataset)
        self.dev_dataset = dev_dataset
        self.meta_dataset = meta_dataset
        self.test_dataset = test_dataset
        self.total_dev_dataset = total_dev_dataset
        self.unlabel_dataset = unlabel_dataset
        self.best_dev_loss_ls = []
        self.best_dev_acc_ls = []
        self.best_dev_loss = float('inf')
        self.best_dev_acc = -1
        self.best_test_acc_ls = []
        self.best_test_acc = -1
        self.global_best_epoch_ls = []
        self.global_best_epoch = None
        if self.config.pretrained_model == "bert":
            collate_fn_ = customize_collate_fn  
        elif self.config.pretrained_model == "distill":
            collate_fn_ = customize_collate_fn_distill
        elif self.config.pretrained_model == "distill2":
            collate_fn_ = customize_collate_fn_distill2
        self.train_loader = DataLoader(
                            dataset=self.train_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.train_dataset,
                                mode='train',
                                max_bsz=self.config.train_batch_size,
                                max_token_len=self.config.train_max_token_len,
                                shuffle = True),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.meta_loader = DataLoader(
                            dataset=self.meta_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.meta_dataset,
                                mode='train',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = True),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.unlabel_loader = DataLoader(
                            dataset=self.unlabel_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.unlabel_dataset,
                                mode='train',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = True),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.valid_loader = DataLoader(
                            dataset=self.dev_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.dev_dataset,
                                mode='dev',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = False),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.total_valid_loader = DataLoader(
                            dataset=self.total_dev_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.total_dev_dataset,
                                mode='dev',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = False),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.test_loader = DataLoader(
                            dataset=self.test_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.test_dataset,
                                mode='dev',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = False),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.label_mapping = label_mapping
        self.max_uncertainty = -np.log(1.0/float(len(self.label_mapping)))
        
        self.early_stopping = None
        
        self.save_path = save_path
        self.sup_path = self.save_path +'/sup'
        self.ssl_path = self.save_path +'/ssl'
        
        self.similarity_standard = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        self.debug = False
        self.debug_sample_thres = 50
        if self.debug:
            self.config.hist_len -= 5 #to speed up
        

        if not os.path.isabs(self.sup_path):
            self.sup_path = os.path.join(os.getcwd(), self.sup_path)
        if not os.path.exists(self.sup_path):
            os.makedirs(self.sup_path)
        
        if not os.path.isabs(self.ssl_path):
            self.ssl_path = os.path.join(os.getcwd(), self.ssl_path)
        if not os.path.exists(self.ssl_path):
            os.makedirs(self.ssl_path)
        self.init_meta()

        
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
        
    def train_epoch(self, epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        
        for idx, (src,targets,inds,ori_targets) in enumerate(self.train_loader):
            if self.debug and idx >= self.debug_sample_thres:
                break
            if self.config.pretrained_model == "bert":
                ids,attention_mask,token_type_ids = self.get_inputs(src)
            else:
                ids,attention_mask = self.get_inputs(src)
            targets = targets.to(self.device, dtype=torch.long)
            if self.config.pretrained_model == "bert":
                outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
            else:
                outputs = self.model(ids, attention_mask, labels=targets)
            loss, logits = outputs[0], outputs[1] # opt
            # attn = None
            
            # if self.model_type != 'baseline':
                # attn = outputs[2]
                # self.build_lexicon(ids, targets, attn)
            
            tr_loss += loss.item()
            scores = torch.softmax(logits, dim=-1)
            big_val, big_idx = torch.max(scores.data, dim=-1)
            n_correct += self.calculate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # if idx % 2 == 0:
            #     dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.total_valid_loader)
            #     self.logger.info(f"Training Epoch {epoch},batch {idx},Dev Loss {dev_loss},Dev Acc is {dev_acc}")

        self.logger.info(f"max # batches are {idx + 1}")
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct)/nb_tr_examples
        self.logger.info(f"Training Epoch {epoch},Training Loss: {epoch_loss}, Training ACC: {epoch_accu}")
        
    def train(self,total_epochs,current_outer_epoch,save=True,early_stop=False,need_test = False,need_break = False):
        if self.config.main_target == "acc":
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True,mode='-')
        else:
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True,mode='-')
        # self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        self.early_stop_epoch = self.config.hist_len
        # if save:
        #     torch.save({'model_state_dict':self.model.state_dict(),
        #                 'optimizer_state_dict':self.optimizer.state_dict(),'epoch':-1},
        #                    self.sup_path +'/checkpoint_-1.pt')
        for epoch in range(total_epochs):
            self.train_epoch(epoch)
            dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.total_valid_loader) # change
            self.logger.info(f"Training Epoch {epoch},Dev Loss {dev_loss},Dev Acc is {dev_acc}")
            
            if self.config.main_target == "loss" and self.best_dev_loss > dev_loss and not self.early_stopping.early_stop: 
                self.best_dev_loss = dev_loss
                self.best_dev_acc = dev_acc
                self.global_best_model = deepcopy(self.model)
                self.global_best_epoch = (current_outer_epoch,epoch)
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/global_best.pt')
                if need_test and not self.debug:
                    test_loss, self.best_test_acc, pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
            if self.config.main_target == "acc" and self.best_dev_acc < dev_acc and not self.early_stopping.early_stop:
                self.best_dev_loss = dev_loss
                self.best_dev_acc = dev_acc
                self.global_best_model = deepcopy(self.model)
                self.global_best_epoch = (current_outer_epoch,epoch)
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/global_best.pt')
                                   
                    
                if need_test and not self.debug:
                    test_loss, self.best_test_acc, pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
            if need_test and not self.debug and epoch % 5 == 0:
                tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
                self.logger.info(f"Regualarly Test Epoch (outer {current_outer_epoch} inner {epoch}),tmp_test_loss {tmp_test_loss}, tmp_test_ACC is {tmp_test_acc}")
            if save:
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/checkpoint_{epoch}.pt')
            if early_stop and self.early_stop_model is None:
                if self.config.main_target == "acc":
                    self.early_stopping(dev_loss,self.logger) # dev_acc
                else:
                    self.early_stopping(dev_loss,self.logger)
                # self.early_stopping(dev_loss,self.logger)
                if self.early_stopping.early_stop:
                    self.logger.info("Early Stopping!")
                    self.early_stop_model = deepcopy(self.model)
                    self.early_stop_epoch = epoch + 1
                    if need_break:
                        break
        
        self.best_dev_loss_ls.append(self.best_dev_loss)
        self.best_dev_acc_ls.append(self.best_dev_acc)
        self.best_test_acc_ls.append(self.best_test_acc)
        self.global_best_model_ls.append(self.global_best_model)
        self.global_best_epoch_ls.append(self.global_best_epoch)
        self.logger.info(f"train done, global Best Epoch (outer {self.global_best_epoch[0]} inner {self.global_best_epoch[1]}),global Best Dev Loss is {self.best_dev_loss}, corresponding global Best Dev Acc is {self.best_dev_acc},corresponding global Best Test Acc is {self.best_test_acc}")
                
    
    def co_teaching(self,early_stop=True):
        # Build tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        # Criterion & optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate) #or AdamW
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.config.learning_rate)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=self.config.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, self.config.max_step, eta_min=self.config.lr_min)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, self.config.max_step, eta_min=self.config.lr_min)
        rate_scheduler = torch.ones(self.config.max_step // self.config.forget_schedule) * self.config.forget_rate
        rate_scheduler[:self.config.T_k] = torch.linspace(0, self.config.forget_rate, self.config.T_k)
        train_step = 0
        log_loss1, log_loss2 = 0.0, 0.0
        best_val_score = None
        log_start_time = time.time()
        
        model = model.to(self.device)
        model1 = model1.to(self.device)
        model2 = model2.to(self.device)
        model1 = DataParallel(model1)
        model2 = DataParallel(model2)

        model.train()
        self.train(self.config.hist_len,0,save=True,early_stop=early_stop,need_test=True,need_break=True)
        self.logger.info("Finish Supervised Training!")
        self.logger.info("#"*100)
        self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
        self.pseudo_labeling(self.unlabel_loader, None)
        selected_LD = self.select(threshold=0.0,epoch=0,type="naive")
        combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD)
        self.train_loader.dataset._data = deepcopy(combined_dataset._data)


        del model
        torch.cuda.empty_cache() 
        self.logger.info("Begin Co-teaching")
        self.logger.info("#"*100)

        best_round = 0
        for epoch in range(0, self.config.co_epochs):
            model1.train()
            model2.train()
            batch_idx = 0
            log_loss1, log_loss2 = 0.0, 0.0
            for idx, (src,targets,inds,ori_targets) in enumerate(self.train_loader):
                ids,attention_mask,token_type_ids = self.get_inputs(src)
                targets = targets.to(self.device, dtype=torch.long)
                ori_targets = ori_targets.to(self.device, dtype=torch.long)
                outputs1 = model1(ids, attention_mask, token_type_ids, labels=targets)
                outputs2 = model2(ids, attention_mask, token_type_ids, labels=targets)
                loss1, logits1 = outputs1[0], outputs1[1]
                loss2, logits2 = outputs2[0], outputs2[1]
                scores1 = torch.softmax(logits1, dim=-1)
                scores2 = torch.softmax(logits2, dim=-1)
                log_prob1 = F.log_softmax(logits1, dim=-1)
                log_prob2 = F.log_softmax(logits2, dim=-1)
                big_val, pred1 = torch.max(scores1.data, dim=-1)
                big_val, pred2 = torch.max(scores2.data, dim=-1)
                disagree_indice = (pred1 != pred2).nonzero(as_tuple=False).flatten()
                if disagree_indice.shape[0]==0:
                    disagree_indice = torch.arange(0, pred1.shape[0])
                loss1_sample = self.cross_entropy_sample(logits1,targets)
                loss2_sample = self.cross_entropy_sample(logits2,targets)

                s_loss1, indice1 = torch.sort(loss1_sample)
                s_loss2, indice2 = torch.sort(loss2_sample)
                if self.config.co_teaching_plus:
                    loss1_sample = loss1_sample[disagree_indice]
                    loss2_sample = loss2_sample[disagree_indice]
                remember_rate = 1 - rate_scheduler[train_step // self.config.forget_schedule]
                # remember_rate = 1 - min(self.config.forget_rate, train_step / self.config.T_k * self.config.forget_rate)
                num_remember = int(remember_rate * len(s_loss1))

                indice1 = indice1[:num_remember]
                indice2 = indice2[:num_remember]

                loss1_from_peer = F.nll_loss(log_prob1[indice2],targets[indice2])
                loss2_from_peer = F.nll_loss(log_prob2[indice1],targets[indice1])
                
                loss1_from_peer.backward()
                loss2_from_peer.backward()
                log_loss1 += loss1_from_peer.item()
                log_loss2 += loss2_from_peer.item()

                train_step += 1
                # if C['warmup_step'] > 0 and train_step < C['warmup_step']:
                #     optimizer1.param_groups[0]['lr'] = C['lr'] * train_step / C['warmup_step']
                #     optimizer2.param_groups[0]['lr'] = C['lr'] * train_step / C['warmup_step']
                torch.nn.utils.clip_grad_norm_(model1.parameters(), 0.2)
                torch.nn.utils.clip_grad_norm_(model2.parameters(), 0.2)
                optimizer1.step()
                optimizer2.step()
                # if train_step >= C['warmup_step']:
                #     scheduler1.step()
                #     scheduler2.step()

                # log & eval
                if train_step % self.config.log_interval == 0:
                    elapsed = time.time() - log_start_time
                    log_str = 'Epoch {:d}, Step {:d}, Batch {:d} | Speed {:.2f}ms/it, LR {:3g} | Loss1 {:.8f}, Loss2 {:.8f}'.format(
                            epoch, train_step, batch_idx,
                            elapsed * 1000 / self.config.log_interval,
                            optimizer1.param_groups[0]['lr'],
                            log_loss1 / self.config.log_interval, log_loss2 / self.config.log_interval)
                    self.logger.info(log_str)
                log_loss1, log_loss2 = 0.0, 0.0
                log_start_time = time.time()


                nb_eval_steps = 0
                nb_eval_examples = 0
                n_correct1, n_correct2, n_total = 0, 0, 0

                if train_step % self.config.eval_interval==0:
                    model1.eval()
                    model2.eval()
                    total_err1, total_err2, total_len = 0, 0, 0
                    eval_start_time = time.time()
                    with torch.no_grad():
                        for idx, (src,targets,inds,ori_targets) in enumerate(self.total_valid_loader):
                            ids,attention_mask,token_type_ids = self.get_inputs(src)
                            targets = targets.to(self.device, dtype=torch.long)
                            ori_targets = ori_targets.to(self.device, dtype=torch.long)
                            outputs1 = model1(ids, attention_mask, token_type_ids, labels=ori_targets)
                            outputs2 = model2(ids, attention_mask, token_type_ids, labels=ori_targets)
                            loss1, logits1 = outputs1[0], outputs1[1]
                            loss2, logits2 = outputs2[0], outputs2[1]
                            scores1 = torch.softmax(logits1, dim=-1)
                            scores2 = torch.softmax(logits2, dim=-1)
                            log_prob1 = F.log_softmax(logits1, dim=-1)
                            log_prob2 = F.log_softmax(logits2, dim=-1)
                            big_val, pred1 = torch.max(scores1.data, dim=-1)
                            big_val, pred2 = torch.max(scores2.data, dim=-1)
                            n_correct1 += self.calculate_accu(pred1,ori_targets)
                            n_correct2 += self.calculate_accu(pred2,ori_targets)
                            nb_eval_examples += targets.size(0)
                    acc1 = (n_correct1 / nb_eval_examples)*100.0
                    acc2 = (n_correct2 / nb_eval_examples)*100.0
                    log_str = 'Eval {:d} at Step {:d} | Finish within {:.2f}s | ACC1 {:.4f}% ACC2 {:.4f}%'.format(
                        train_step // self.config.eval_interval - 1, train_step, time.time() - eval_start_time, acc1, acc2)
                    self.logger.info(log_str)
                    (acc,model_best) = (acc1, model1) if acc1 > acc2 else (acc2, model2)
                    if best_val_score is None or acc >= best_val_score:
                        torch.save({'model_state_dict':model_best.state_dict(),
                                            'optimizer_state_dict_1':optimizer1.state_dict(),'epoch':epoch,
                                            'optimizer_state_dict_2':optimizer2.state_dict(),'epoch':epoch},
                                        self.sup_path +f'/checkpoint_{epoch}.pt')
                        best_val_score = acc
                        self.global_best_model = deepcopy(model_best)
                        best_round = train_step // self.config.eval_interval - 1
                        self.logger.info(f'Better Val ACC detected:{acc}, model saved...')
                        tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.global_best_model, self.test_loader, is_test=True, return_details=True)
                        self.logger.info(f"The related Test: Test Loss {tmp_test_loss}, Test ACC is {tmp_test_acc}")
                model1.train()
                model2.train()

        self.logger.info(f'Global best ACC detected:{best_val_score}, at round:{best_round}')
        tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.global_best_model, self.test_loader, is_test=True, return_details=True)


    def SELF(self,early_stop=True):
        # hyper & init
        alpha = 0.1
        
        # self.filtered_dataset = deepcopy(self.train_dataset)

        # Build tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        # Criterion & optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate) #or AdamW
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.config.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, self.config.max_step, eta_min=self.config.lr_min)

        train_step = 0
        log_loss1, log_loss2 = 0.0, 0.0
        best_val_score = None
        log_start_time = time.time()
        
        model = model.to(self.device)
        model1 = model1.to(self.device)
        model1 = DataParallel(model1)
        

        model.train()
        self.train(self.config.hist_len,0,save=True,early_stop=early_stop,need_test=True,need_break=True)
        self.logger.info("Finish Supervised Training!")
        self.logger.info("#"*100)
        self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
        self.pseudo_labeling(self.unlabel_loader, None)
        selected_LD = self.select(threshold=0.0,epoch=0,type="naive")
        combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD)
        self.train_loader.dataset._data = deepcopy(combined_dataset._data)


        del model
        torch.cuda.empty_cache() 
        self.logger.info("Begin SELF")
        self.logger.info("#"*100)

        # init for SELF

        self.train_hist_prob = np.zeros(shape=(len(self.train_loader.dataset),self.config.class_num)).astype(np.float32)
        self.global_filter_masking = torch.ones(len(self.train_loader.dataset))
        

        for epoch in range(0, 20):
            model1.train()
            batch_idx = 0
            log_loss1 = 0.0
            for idx, (src,targets,inds,ori_targets) in enumerate(self.train_loader):
                ids,attention_mask,token_type_ids = self.get_inputs(src)
                targets = targets.to(self.device, dtype=torch.long)
                ori_targets = ori_targets.to(self.device, dtype=torch.long)
                outputs1 = model1(ids, attention_mask, token_type_ids, labels=targets)
                loss1, logits1 = outputs1[0], outputs1[1]
                scores1 = torch.softmax(logits1, dim=-1)
                log_prob1 = F.log_softmax(logits1, dim=-1)
                big_val, pred1 = torch.max(scores1.data, dim=-1)
                loss1_sample = self.cross_entropy_sample(logits1,targets)
                # Filtering the data
                loss1_sample_mask = loss1_sample * self.global_filter_masking[inds].to(self.device)
                loss1_sample_mask.mean().backward()
                # loss1.backward()

                train_step += 1
                # if C['warmup_step'] > 0 and train_step < C['warmup_step']:
                #     optimizer1.param_groups[0]['lr'] = C['lr'] * train_step / C['warmup_step']
                #     optimizer2.param_groups[0]['lr'] = C['lr'] * train_step / C['warmup_step']
                torch.nn.utils.clip_grad_norm_(model1.parameters(), 0.2)
                optimizer1.step()
                # if train_step >= C['warmup_step']:
                #     scheduler1.step()
                #     scheduler2.step()

                # log & eval
                if train_step % self.config.log_interval == 0:
                    elapsed = time.time() - log_start_time
                    log_str = 'Epoch {:d}, Step {:d}, Batch {:d} | Speed {:.2f}ms/it, LR {:3g} | Loss1 {:.8f}'.format(
                            epoch, train_step, batch_idx,
                            elapsed * 1000 / self.config.log_interval,
                            optimizer1.param_groups[0]['lr'],
                            log_loss1 / self.config.log_interval)
                    self.logger.info(log_str)
                log_loss1 = 0.0
                log_start_time = time.time()


                nb_eval_steps = 0
                nb_eval_examples = 0
                n_correct1, n_total = 0, 0

                if train_step % self.config.eval_interval==0:
                    model1.eval()
                    total_err1, total_len = 0, 0
                    eval_start_time = time.time()
                    with torch.no_grad():
                        for idx, (src,targets,inds,ori_targets) in enumerate(self.total_valid_loader):
                            ids,attention_mask,token_type_ids = self.get_inputs(src)
                            targets = targets.to(self.device, dtype=torch.long)
                            ori_targets = ori_targets.to(self.device, dtype=torch.long)
                            outputs1 = model1(ids, attention_mask, token_type_ids, labels=ori_targets)
                            loss1, logits1 = outputs1[0], outputs1[1]
                            scores1 = torch.softmax(logits1, dim=-1)
                            log_prob1 = F.log_softmax(logits1, dim=-1)
                            big_val, pred1 = torch.max(scores1.data, dim=-1)
                            n_correct1 += self.calculate_accu(pred1,ori_targets)
                            nb_eval_examples += targets.size(0)
                    acc1 = (n_correct1 / nb_eval_examples)*100.0
                    log_str = 'Eval {:d} at Step {:d} | Finish within {:.2f}s | ACC1 {:.4f}%'.format(
                        train_step // self.config.eval_interval - 1, train_step, time.time() - eval_start_time, acc1)
                    self.logger.info(log_str)
                    (acc,model_best) = (acc1, model1)
                    if best_val_score is None or acc >= best_val_score:
                        torch.save({'model_state_dict':model_best.state_dict(),
                                            'optimizer_state_dict_1':optimizer1.state_dict(),'epoch':epoch},
                                        self.sup_path +f'/checkpoint_{epoch}.pt')
                        best_val_score = acc
                        self.global_best_model = deepcopy(model_best)
                        best_round = train_step // self.config.eval_interval - 1
                        self.logger.info(f'Better Val ACC detected:{acc}, model saved...')
                        tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.global_best_model, self.test_loader, is_test=True, return_details=True)
                        self.logger.info(f"The related Test: Test Loss {tmp_test_loss}, Test ACC is {tmp_test_acc}")
                model1.train()
            
            # Infer on train data using the best model and filter out data
            model1 = deepcopy(self.global_best_model)
            model1.eval()
            cur_pred_prob = np.zeros(shape=(len(self.train_loader.dataset),self.config.class_num)).astype(np.float32)

            # Filter
            with torch.no_grad():
                for idx, (src,targets,inds,ori_targets) in enumerate(self.train_loader):
                    ids,attention_mask,token_type_ids = self.get_inputs(src)
                    ori_targets = ori_targets.to(self.device,dtype=torch.long)
                    outputs1 = model1(ids, attention_mask, token_type_ids, labels=ori_targets)
                    loss1, logits1 = outputs1[0], outputs1[1]
                    pred_prob = torch.softmax(logits1, dim=-1)
                    ensemble_pred_prob = alpha * self.train_hist_prob[inds] + (1-alpha) * pred_prob.cpu().detach().numpy()
                    big_val, ens_pred = torch.max(torch.tensor(ensemble_pred_prob), dim=-1)
                    correct = ori_targets.cpu() == ens_pred
                    inds_list = inds[correct == 0].cpu().tolist()

                    self.global_filter_masking[inds_list] = 0 # update the filtering mask
                    self.train_hist_prob[inds] = ensemble_pred_prob # update the historical probs
            # n_select = 0
            # select_data = {}
            # for i,ind in enumerate(inds_list):
            #     select_data[ind] = deepcopy(self.train_loader.dataset._data[ind])
            #     n_select += 1
            # self.filtered_dataset, _ = self.add_dataset(self.filtered_dataset, select_data)
            # self.train_loader.dataset._data = deepcopy(self.filtered_dataset._data)
        self.logger.info(f'Epoch:{epoch}, Global best Val ACC detected:{best_val_score}')

    def l2r(self,early_stop=True):
        # hyper & init
        alpha = 0.1
        
        # self.filtered_dataset = deepcopy(self.train_dataset)

        # Build tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        # Criterion & optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate) #or AdamW
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.config.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, self.config.max_step, eta_min=self.config.lr_min)

        train_step = 0
        log_loss, log_meta_loss = 0.0, 0.0
        best_val_score = None
        log_start_time = time.time()
        
        model = model.to(self.device)
        # model1 = DataParallel(model1)
        model1 = model1.to(self.device)

        model.train()
        self.train(self.config.hist_len,0,save=True,early_stop=early_stop,need_test=True,need_break=True)
        self.logger.info("Finish Supervised Training!")
        self.logger.info("#"*100)
        self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
        self.pseudo_labeling(self.unlabel_loader, None)
        selected_LD = self.select(threshold=0.85,epoch=0,type="naive")
        combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD)
        self.train_loader.dataset._data = deepcopy(combined_dataset._data)
        clean_data_iter = iter(self.total_valid_loader)


        del model
        del optimizer
        torch.cuda.empty_cache() 
        self.logger.info("Begin L2R")
        self.logger.info("#"*100)
        log_start_time = time.time() 


        for epoch in range(0, 20):
            model1.train()
            batch_idx = 0
            log_loss1 = 0.0
            for idx, (src,targets,inds,ori_targets) in enumerate(self.train_loader):
                # model1 = DataParallel(model1)
                # model1 = model1.to(self.device)
                # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                # self.device_meta = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                # fmodel = fmodel.to(self.device_meta)
                # fmodel = DataParallel(fmodel)
                ids,attention_mask,token_type_ids = self.get_inputs(src)
                targets = targets.to(self.device, dtype=torch.long)
                # ori_targets = ori_targets.to(self.device, dtype=torch.long)
                with higher.innerloop_ctx(model1, optimizer1, copy_initial_weights=False) as (fmodel, diffopt):
                    outputs1 = fmodel(ids, attention_mask, token_type_ids, labels=targets)
                    loss, meta_logit = outputs1[0], outputs1[1]
                    meta_log_prob = F.log_softmax(meta_logit,dim=-1)
                    cost = F.nll_loss(meta_log_prob,targets,reduction='none')
                    eps = torch.zeros_like(cost, requires_grad=True, device=self.device)
                    meta_loss = torch.sum(cost*eps)
                    diffopt.step(meta_loss)
                    # diffopt.step(meta_loss, grad_callback=grad_clipping)
                    # fmodel.zero_grad()
                    # params = gradient_update_parameters(model1, meta_loss, step_size=optimizer.param_groups[0]['lr'])
                    try:
                        src_c,targets_c,inds_c,ori_targets_c = next(clean_data_iter)
                    except StopIteration:
                        clean_data_iter = iter(self.total_valid_loader)
                        src_c,targets_c,inds_c,ori_targets_c = next(clean_data_iter)
                    ids_c,attention_mask_c,token_type_ids_c = self.get_inputs(src_c)
                    targets_c = targets_c.to(self.device,dtype=torch.long)
                    outputs_c = fmodel(ids_c,attention_mask_c,token_type_ids_c,labels=targets_c)
                    clean_loss, clean_logit = outputs_c[0], outputs_c[1]
                    log_meta_loss += clean_loss.item()
                    grad_eps = torch.autograd.grad(clean_loss, eps, only_inputs=True)[0]

                    w = torch.clamp(-grad_eps,min=0)
                    if torch.sum(w)!=0:
                        w=w/torch.sum(w)
                
                output = model1(ids, attention_mask, token_type_ids, labels=targets)
                loss, logit = output[0], output[1]
                log_prob = F.log_softmax(logit,dim=-1)
                cost = F.nll_loss(log_prob,targets,reduction='none')
                loss = torch.sum(cost*w)
                log_loss += loss.item()
                optimizer1.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model1.parameters(),0.25)
                optimizer1.step()
                torch.cuda.empty_cache()
                # if train_step >= self.config.warmup_step:
                #     scheduler.step()
                train_step += 1

                # log & eval
                if train_step % self.config.log_interval == 0:
                    elapsed = time.time() - log_start_time
                    log_str = 'Epoch {:d}, Step {:d}, Batch {:d} | Speed {:.2f}ms/it, LR {:3g} | Loss {:.8f} ｜ Meta Loss {:.8f}'.format(
                            epoch, train_step, batch_idx,
                            elapsed * 1000 / self.config.log_interval,
                            optimizer1.param_groups[0]['lr'],
                            log_loss / self.config.log_interval, log_meta_loss / self.config.log_interval)
                    self.logger.info(log_str)
                log_loss, log_meta_loss = 0.0, 0.0
                log_start_time = time.time()


                nb_eval_steps = 0
                nb_eval_examples = 0
                n_correct1, n_total = 0, 0

                if train_step % self.config.eval_interval==0:
                    model1.eval()
                    total_err1, total_len = 0, 0
                    eval_start_time = time.time()
                    with torch.no_grad():
                        for idx, (src,targets,inds,ori_targets) in enumerate(self.total_valid_loader):
                            ids,attention_mask,token_type_ids = self.get_inputs(src)
                            # targets = targets.to(self.device, dtype=torch.long)
                            ori_targets = ori_targets.to(self.device, dtype=torch.long)
                            outputs1 = model1(ids, attention_mask, token_type_ids, labels=ori_targets)
                            loss1, logits1 = outputs1[0], outputs1[1]
                            scores1 = torch.softmax(logits1, dim=-1)
                            log_prob1 = F.log_softmax(logits1, dim=-1)
                            big_val, pred1 = torch.max(scores1.data, dim=-1)
                            n_correct1 += self.calculate_accu(pred1,ori_targets)
                            nb_eval_examples += ori_targets.size(0)
                    acc1 = (n_correct1 / nb_eval_examples)*100.0
                    log_str = 'Eval {:d} at Step {:d} | Finish within {:.2f}s | ACC1 {:.4f}%'.format(
                        train_step // self.config.eval_interval - 1, train_step, time.time() - eval_start_time, acc1)
                    self.logger.info(log_str)
                    (acc,model_best) = (acc1, model1)
                    if best_val_score is None or acc >= best_val_score:
                        torch.save({'model_state_dict':model_best.state_dict(),
                                            'optimizer_state_dict_1':optimizer1.state_dict(),'epoch':epoch},
                                        self.sup_path +f'/checkpoint_{epoch}.pt')
                        best_val_score = acc
                        self.global_best_model = deepcopy(model_best)
                        best_round = train_step // self.config.eval_interval - 1
                        self.logger.info(f'Better Val ACC detected:{acc}, model saved...')
                        tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.global_best_model, self.test_loader, is_test=True, return_details=True)
                        self.logger.info(f"The related Test: Test Loss {tmp_test_loss}, Test ACC is {tmp_test_acc}")
                    



    def self_train(self,load=False,early_stop=True,check_signal=False,load_path=None):
        selected_LD = None
        for outer_epoch in range(self.config.epochs):

            self.model.train() # 0
            self.init_meta() #init meta from scratch
            if selected_LD is None and load:
                #load pretrained model
                self.load(os.path.join(self.sup_path,f"checkpoint_{self.config.hist_len - 1}.pt"))
                self.global_best_model = deepcopy(self.model)
            else:
                #combine selected unlabel and labeled            
                combined_dataset,combined_meta = self.add_dataset(self.label_dataset,selected_LD,self.config.add_meta_ratio,self.meta_dataset)
                self.train_loader.dataset._data = deepcopy(combined_dataset._data)
                if combined_meta is not None:
                    self.meta_loader.dataset._data = deepcopy(combined_meta._data)
                self.logger.info(f'updated train set size {len(self.train_loader.dataset)}, updated meta set size {len(self.meta_loader.dataset)}')
                if self.config.training_from_scratch == True:
                    self.init_main(outer_epoch) 
                #train on combined dataset, in first epoch, this is only the label dataset
                if early_stop and self.early_stop_model is not None:
                    self.model = deepcopy(self.early_stop_model)
                    self.early_stop_model = None
                self.train(self.config.hist_len,outer_epoch,save=True,early_stop=early_stop,need_test=True,need_break=True)
            

            if outer_epoch == self.config.epochs - 1: 
                if self.config.training_from_scratch == True:
                    if self.config.main_target == "loss":
                        all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,
                                            self.best_dev_loss_ls,self.best_test_acc_ls))
                        all_info = sorted(all_info, key=lambda x:x[2])
                        self.global_best_model = all_info[0][0]
                        self.global_best_epoch = all_info[0][1]
                        self.best_dev_loss = all_info[0][2]
                        self.best_test_acc = all_info[0][3]
                        self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_loss:{self.best_dev_loss} best_test_acc:{self.best_test_acc}')
                        torch.save({'model_state_dict':self.model.state_dict()},
                                    self.sup_path +f'/checkpoint_best.pt')
                    else:
                        all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,self.best_dev_loss_ls,
                        self.best_dev_acc_ls,self.best_test_acc_ls))
                        all_info = sorted(all_info, key=lambda x:(x[3],-x[2]))
                        self.global_best_model = all_info[-1][0]
                        self.global_best_epoch = all_info[-1][1]
                        self.best_dev_acc = all_info[-1][3]
                        self.best_test_acc = all_info[-1][4]
                        self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_acc:{self.best_dev_acc} best_test_acc:{self.best_test_acc}')
                        torch.save({'model_state_dict':self.model.state_dict()},
                                    self.sup_path +f'/checkpoint_best.pt')
                return
            self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
            self.pseudo_labeling(self.unlabel_loader, None)
            self.logger.info("######### pseudo labeling on META dataset ############")
            self.pseudo_labeling(self.meta_loader, None) #dev 1 for training meta model
            self.logger.info("######### pseudo labeling on DEV dataset ############")
            self.pseudo_labeling(self.valid_loader,None) #dev 2 for evaluate meta model
            # self.total_hist_len = self.early_stop_epoch + 1 
            self.total_hist_len = self.early_stop_epoch
            # self.total_hist_len = self.config.hist_len
            self.init_hist_signals()
            #start infer history signals 
            self.infer_hist(check_signal)
            #after infer history, save history for future debugging and analysis
            # self.save_hist(outer_epoch)
            #after we get history signals, we can start train meta model metaM
            self.train_meta(epoch=outer_epoch)
            selected_LD = self.select(threshold=self.config.meta_cfd_threshold,epoch=outer_epoch)
            if outer_epoch == 0:
                self.logger.info("#"*100)
                self.logger.info("Finish supervised training!")
                self.logger.info("#"*100)
            if outer_epoch == 1 and self.config.block_entropy:
                self.config.use_entropy = False
                self.config.input_size -= 2
                self.evaluator.config = self.config #update config to evaluator

    

    def init_main(self,outer_epoch):
        if outer_epoch == 0:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        else: # noisy student
            # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,hidden_dropout_prob=self.config.dropout,output_hidden_states=True)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.config.hist_len * (len(self.train_dataset) // self.config.train_batch_size), eta_min=self.config.learning_rate * 0.01,verbose=False)
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.best_dev_loss = float('inf')
        self.best_dev_acc = -1
        self.best_test_acc = -1
        self.global_best_epoch = None
        self.early_stopping = None
        self.global_best_model = None
        self.best_meta_model = None
        self.best_meta_thresh = None
        self.early_stop_model = None
        self.early_stop_epoch = self.config.hist_len

    def init_meta(self):
        self.meta_model = MLP_head(self.config.hidden_size,self.config.input_size,output_size=2,act=self.config.act).to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.config.meta_lr)
        self.meta_ce_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.meta_optimizer,self.config.meta_epochs*self.config.multiple_len, eta_min=self.config.meta_lr_min)
        self.meta_loader.dataset._data = deepcopy(self.meta_dataset._data)
        print(f"reinitlize meta data loader with dataset size as {len(self.meta_loader.dataset)}")
    
    def init_hist_signals(self):
        init_len = self.total_hist_len
        self.logger.info("initalize history signals")
        self.unlabel_hist_loss = np.zeros(shape=(len(self.unlabel_loader.dataset),init_len)).astype(np.float32)
        self.unlabel_hist_track = np.zeros(shape=(len(self.unlabel_loader.dataset),init_len)).astype(np.int32)
        self.unlabel_instant_loss = np.zeros(shape=(len(self.unlabel_loader.dataset),init_len)).astype(np.float32)
        self.unlabel_instant_simi = np.zeros(shape=(len(self.unlabel_loader.dataset),init_len)).astype(np.float32)
        self.unlabel_hist_prob = np.zeros(shape=(len(self.unlabel_loader.dataset),init_len,self.config.class_num)).astype(np.float32)

        self.meta_hist_loss = np.zeros(shape=(len(self.meta_loader.dataset),init_len)).astype(np.float32)
        self.meta_hist_track = np.zeros(shape=(len(self.meta_loader.dataset),init_len)).astype(np.int32)
        self.meta_instant_loss = np.zeros(shape=(len(self.meta_loader.dataset),init_len)).astype(np.float32)
        self.meta_instant_simi = np.zeros(shape=(len(self.meta_loader.dataset),init_len)).astype(np.float32)
        self.meta_hist_prob = np.zeros(shape=(len(self.meta_loader.dataset),init_len,self.config.class_num)).astype(np.float32)

        self.dev_hist_loss = np.zeros(shape=(len(self.valid_loader.dataset),init_len)).astype(np.float32)
        self.dev_hist_track = np.zeros(shape=(len(self.valid_loader.dataset),init_len)).astype(np.int32)
        self.dev_instant_loss = np.zeros(shape=(len(self.valid_loader.dataset),init_len)).astype(np.float32)
        self.dev_instant_simi = np.zeros(shape=(len(self.valid_loader.dataset),init_len)).astype(np.float32)
        self.dev_hist_prob = np.zeros(shape=(len(self.valid_loader.dataset),init_len,self.config.class_num)).astype(np.float32)

        self.logger.info(f"unlabel_loader len {len(self.unlabel_loader.dataset)}, meta_loader len {len(self.meta_loader.dataset)}, valid_loader len {len(self.valid_loader.dataset)}")


    def check_signal(self,all_signals,sifts):
        #check unlabel signals
        res = {"correct":[],"incorrect":[]}
        for signal_idx,signals in enumerate(all_signals):
            crct_res = []
            incrct_res = []
            for batch_idx,signal in enumerate(signals):
                sift = sifts[batch_idx]
                crct_sigs = signal[sift].squeeze().tolist()
                incrct_sigs = signal[~sift].squeeze().tolist()
                crct_res.extend([crct_sigs] if isinstance(crct_sigs,float) else crct_sigs)
                incrct_res.extend([incrct_sigs] if isinstance(incrct_sigs,float) else incrct_sigs)
            res['correct'].append(crct_res)
            res['incorrect'].append(incrct_res)
        return res

    def naive_self_train(self,load=False,early_stop=True):
        best_accuracy = -1
        selected_LD = None
        for outer_epoch in range(self.config.epochs):
            self.model.train()
            if selected_LD is None and load:
                self.load("/mfs/newresearch/self-training/ours/output_youtube/0_baseline_20211027-094959/sup/global_best.pt")
            else:
                #combine selected unlabel and labeled            
                combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD,0,None)
                self.train_loader.dataset._data = deepcopy(combined_dataset._data)
                #train on combined dataset, in first epoch, this is only the label dataset
                if early_stop and self.early_stop_model is not None:
                    self.model = deepcopy(self.early_stop_model)
                if self.config.training_from_scratch == True:
                    self.init_main(outer_epoch) 
                self.train(self.config.hist_len,outer_epoch,save=True,early_stop=early_stop,need_test=True,need_break=True)
            self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
            self.pseudo_labeling(self.unlabel_loader,None)
            # self.logger.info("######### pseudo labeling on DEV dataset ############")
            # self.pseudo_labeling(self.valid_loader,None) #dev 2 for evaluate meta model
            selected_LD = self.select(threshold=self.config.confidence_thres,type="naive") # 0.9
        if self.config.training_from_scratch == True:
            if self.config.main_target == "loss":
                all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,
                                    self.best_dev_loss_ls,self.best_test_acc_ls))
                all_info = sorted(all_info, key=lambda x:x[2])
                self.global_best_model = all_info[0][0]
                self.global_best_epoch = all_info[0][1]
                self.best_dev_loss = all_info[0][2]
                self.best_test_acc = all_info[0][3]
                self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_loss:{self.best_dev_loss} best_test_acc:{self.best_test_acc}')
                torch.save({'model_state_dict':self.model.state_dict()},
                                    self.sup_path +f'/checkpoint_best.pt')
            else:
                all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,self.best_dev_loss_ls,
                self.best_dev_acc_ls,self.best_test_acc_ls))
                all_info = sorted(all_info, key=lambda x:(x[3],-x[2]))
                self.global_best_model = all_info[-1][0]
                self.global_best_epoch = all_info[-1][1]
                self.best_dev_acc = all_info[-1][3]
                self.best_test_acc = all_info[-1][4]
                self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_acc:{self.best_dev_acc} best_test_acc:{self.best_test_acc}')
                torch.save({'model_state_dict':self.model.state_dict()},
                            self.sup_path +f'/checkpoint_best.pt')
        else:
            self.logger.info('Best accuracy {}'.format(best_accuracy))
    
    def thres_self_train(self,load=False,early_stop=True):
        best_accuracy = -1
        selected_LD = None
        for outer_epoch in range(self.config.epochs):
            if self.config.training_from_scratch == True:
                self.init_main(outer_epoch) 
            self.model.train()
            if selected_LD is None and load:
                self.load("/mfs/newresearch/self-training/ours/output_youtube/0_baseline_20211027-094959/sup/global_best.pt")
            else:
                #combine selected unlabel and labeled            
                combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD,0,None)
                self.train_loader.dataset._data = deepcopy(combined_dataset._data)
                #train on combined dataset, in first epoch, this is only the label dataset
                if early_stop and self.early_stop_model is not None: 
                    self.model = deepcopy(self.early_stop_model)
                self.train(self.config.hist_len,outer_epoch,save=True,early_stop=early_stop,need_test=True,need_break=True)
            self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
            self.pseudo_labeling(self.unlabel_loader,None)
            self.logger.info("######### pseudo labeling on DEV dataset ############")
            self.pseudo_labeling(self.valid_loader,None) #dev 2 for evaluate meta model
            self.logger.info("######### Select threshold based on confidence on DEV dataset ############")
            threshold_pick = self.select_threshold()

            selected_LD = self.select(threshold=threshold_pick,type="thres") # 0.9
        if self.config.training_from_scratch == True:
            if self.config.main_target == "loss":
                all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,
                                    self.best_dev_loss_ls,self.best_test_acc_ls))
                all_info = sorted(all_info, key=lambda x:x[2])
                self.global_best_model = all_info[0][0]
                self.global_best_epoch = all_info[0][1]
                self.best_dev_loss = all_info[0][2]
                self.best_test_acc = all_info[0][3]
                self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_loss:{self.best_dev_loss} best_test_acc:{self.best_test_acc}')
                torch.save({'model_state_dict':self.model.state_dict()},
                                    self.sup_path +f'/checkpoint_best.pt')
            else:
                all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,self.best_dev_loss_ls,
                self.best_dev_acc_ls,self.best_test_acc_ls))
                all_info = sorted(all_info, key=lambda x:(x[3],-x[2]))
                self.global_best_model = all_info[-1][0]
                self.global_best_epoch = all_info[-1][1]
                self.best_dev_acc = all_info[-1][3]
                self.best_test_acc = all_info[-1][4]
                self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_acc:{self.best_dev_acc} best_test_acc:{self.best_test_acc}')
                torch.save({'model_state_dict':self.model.state_dict()},
                            self.sup_path +f'/checkpoint_best.pt')
        else:
            self.logger.info('Best accuracy {}'.format(best_accuracy))
    def select_threshold(self):
        meta_gt, meta_probs = [], []
        for idx, (src,targets,inds,ori_targets) in enumerate(self.valid_loader):            
            training_targets = targets == ori_targets
            training_targets = training_targets.to(self.device)
            targets = targets.to(self.device)
            if self.config.pretrained_model == "bert":
                ids,attention_mask,token_type_ids = self.get_inputs(src)
            else:
                ids,attention_mask = self.get_inputs(src)
            if self.config.pretrained_model == "bert":
                outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
            else:
                outputs = self.model(ids, attention_mask, labels=targets)
            loss, logits = outputs[0], outputs[1]
            pred = torch.argmax(logits, dim=-1)
            conf, _ = torch.softmax(logits, dim=-1).max(dim=1)
            meta_gt.extend(training_targets.long().cpu().tolist())
            meta_probs.extend(conf.cpu().tolist())

        optimize_target,threshold, precision, recall, f1_score,acc,fx_score,beta = opt_threshold(meta_gt,meta_probs,self.config.meta_target,self.config.beta)
        self.logger.info(f"pick_F1 {f1_score:.4f}, pick_RECALL {recall:.4f}, pick_PRECISION {precision:.4f},pick_FX {fx_score:.4f}, beta {self.config.beta}")
        self.logger.info(f"target metric:{self.config.meta_target} - {optimize_target:.4f}, pick_THRESHOLD {threshold:.2f}")

        return threshold
    def train_meta(self,save=True,early_stop = True,epoch='None'):
        #train meta dataset
        self.model.eval()
        self.meta_model.train()
        best_eval_target = -1
        best_eval_loss = float('inf')
        best_f1,best_acc,best_precistion,best_recall,best_target = None,None,None,None,None
        #infer on meta set
        # for hist_ind in range(self.config.hist_len - self.config.multiple_len,self.config.hist_len+1):
        for infer_ind in range(self.total_hist_len- self.config.multiple_len,self.total_hist_len):
            hist_ind = infer_ind + 1
            # self.meta_early_stopping = EarlyStopping(patience=self.config.meta_patience, verbose=True,mode='+')
            self.meta_early_stopping = EarlyStopping(patience=self.config.meta_patience, verbose=True,mode='-')
            if self.config.use_entropy:
                hist_entropy,cur_entropy = self.calc_entropy(self.meta_hist_track[:,:hist_ind],type='hard') 
                hist_entropy,cur_entropy = hist_entropy.to(self.device),cur_entropy.to(self.device)
            if self.config.soft_entropy:
                if self.config.soft_entropy_mul:
                    hist_entropy_soft, hist_entropy_soft_mul = self.calc_entropy(self.meta_hist_prob,type='soft')
                    hist_entropy_soft_mul[torch.isinf(hist_entropy_soft_mul)] = 0
                    hist_entropy_soft_mul[torch.isnan(hist_entropy_soft_mul)] = 0
                    hist_entropy_soft_mul = hist_entropy_soft_mul.to(self.device)
                else:
                    hist_entropy_soft, _ = self.calc_entropy(self.meta_hist_prob,type='soft')
                hist_entropy_soft[torch.isinf(hist_entropy_soft)] = 0
                hist_entropy_soft[torch.isnan(hist_entropy_soft)] = 0
                hist_entropy_soft = hist_entropy_soft.to(self.device)
            feature_batch = {}
            cor_batch = {}
            for meta_epoch in range(self.config.meta_epochs):
                n_correct, n_total = 0, 0
                meta_training_loss = 0
                meta_preds,meta_gt,meta_probs = [],[],[]
                changed_num,crct_num = 0,0
                
                for idx, (src,targets,inds,ori_targets) in enumerate(self.meta_loader):
                    if idx >= self.debug_sample_thres and self.debug: 
                        break
                    hist_loss_np,cur_loss_np = self.meta_hist_loss[:,hist_ind-1][inds],self.meta_instant_loss[:,hist_ind-1][inds]
                    if self.config.use_simi:
                        cur_simi_np = self.meta_instant_simi[:,hist_ind-1][inds]
                        cur_simi = torch.from_numpy(cur_simi_np).to(self.device)
                    hist_loss,cur_loss = torch.from_numpy(hist_loss_np).to(self.device),torch.from_numpy(cur_loss_np).to(self.device)
                    # if hist_ind != self.config.hist_len:
                        # targets = torch.from_numpy(self.meta_hist_track[:,hist_ind-1][inds])
                    infered_targets = torch.from_numpy(self.meta_hist_track[:,hist_ind-1][inds])
                    
                    training_targets = infered_targets == ori_targets 
                    training_targets = training_targets.to(self.device)
                    if meta_epoch == 0:
                        changed_num += (infered_targets != ori_targets).sum().item()
                        crct_num += training_targets.sum().item()
                    # input_signals = [hist_loss,cur_loss]
                    input_signals = []
                    if self.config.use_loss: # hist_loss
                        if self.config.use_ema_loss:
                            input_signals = [hist_loss[:,None] if len(hist_loss.shape) == 1 else hist_loss,cur_loss[:,None] if len(cur_loss.shape) == 1 else cur_loss]
                        else:
                            input_signals = [cur_loss[:,None] if len(cur_loss.shape) == 1 else cur_loss]
                    if self.config.use_entropy:
                        if self.config.use_hist_entropy:
                            input_signals.extend([hist_entropy[inds][:,None],cur_entropy[inds][:,None]])
                        else:
                            input_signals.extend([cur_entropy[inds][:,None]])
                    if self.config.soft_entropy:
                        input_signals.extend([hist_entropy_soft[inds][:,None]])
                    if self.config.soft_entropy_mul:
                        for i in range(self.config.class_num):
                            hist_entropy_soft_mul_i = hist_entropy_soft_mul[:,i]
                            input_signals.extend([hist_entropy_soft_mul_i[inds][:,None]])
                    if self.config.use_simi:
                        # input_signals.append(cur_simi)
                        input_signals.append(cur_simi[:,None] if len(cur_simi.shape) == 1 else cur_simi)
                    with torch.no_grad():
                        select_label = training_targets.long()
                        feature = torch.cat(input_signals,dim=-1).float()
                        feature_label = torch.cat([feature,select_label.reshape(-1,1)],dim=-1)
                        cor = np.corrcoef(feature_label.cpu().t())
                        feat_label_cor = cor[-1,:-1]
                        feature_batch[idx] = feature_label
                        cor_batch[idx] = feat_label_cor

                    if self.config.test_single_feat:
                        # feat_name_dict = {"hist_emaloss":0,"cur_loss":1,"hist_entropy":2,"cur_entropy":3,"soft_entropy":4,"cur_simi":5}
                        feat_name_dict = {name:i for i,name in enumerate(self.keys)}
                        input_feature = input_signals[feat_name_dict[self.config.test_single_feat_name]]
                        meta_logits = self.meta_model(input_feature.float())
                    else:
                        meta_logits = self.meta_model(torch.cat(input_signals,dim=-1).float())
                    meta_prob = F.softmax(meta_logits,dim=-1)
                    meta_loss = self.meta_ce_loss(meta_logits,training_targets.long())
                    pred = torch.argmax(meta_logits, dim=-1)
                    n_total += targets.size(0)
                    self.meta_optimizer.zero_grad()
                    meta_training_loss += meta_loss.item()
                    meta_loss.backward()
                    self.meta_optimizer.step()
                    meta_preds.extend(pred.long().cpu().tolist())
                    meta_gt.extend(training_targets.long().cpu().tolist())
                    meta_probs.extend(meta_prob[:,1].cpu().tolist())
                # n_correct = (np.array(meta_gt) == np.array(meta_preds)).sum()
                # precision, recall, f1_score, support = precision_recall_fscore_support(meta_gt, meta_preds, average='binary')
                optimize_target,threshold,precision, recall, f1_score,acc,fx_score,beta = opt_threshold(meta_gt,meta_probs,self.config.meta_target,self.config.beta)
                if save:
                    torch.save({'model_state_dict':self.meta_model.state_dict(),
                        'optimizer_state_dict':self.meta_optimizer.state_dict(),'epoch':meta_epoch},
                        self.ssl_path +f'/meta_checkpoint_{meta_epoch}_{hist_ind}.pt')
                self.meta_scheduler.step()
                # acc = (n_correct)/n_total
                if meta_epoch == 0:
                    self.logger.info(f"hist_ind is {hist_ind}, changed_num is {changed_num}, correct inferred {crct_num}, acc inferred {(crct_num / n_total):.4f}")
                self.logger.info(f"meta_epoch {meta_epoch}, hist_ind {hist_ind}, meta_training_loss {meta_training_loss / (idx+1):.4f},meta_training_acc {acc:.4f}")
                self.logger.info(f"meta_training_F1 {f1_score:.4f}, meta_training_RECALL {recall:.4f}, meta_training_PRECISION {precision:.4f},meta_training_Fxscore {fx_score:.4f},beta {self.config.beta}")
                self.logger.info(f"target meta_training_{self.config.meta_target} {optimize_target:.4f}, meta_training_THRESHOLD {threshold:.2f}")

                meta_eval_loss, meta_eval_accu,meta_eval_precision,meta_eval_recall,meta_eval_f1_score,meta_eval_target,meta_thresh = self.evaluator.evaluate_meta(self.meta_model,self.valid_loader,self.dev_hist_loss,
                                                self.dev_hist_track,self.dev_hist_prob,self.dev_instant_loss,self.dev_instant_simi,self.calc_entropy,self.meta_ce_loss,is_test=False)
                self.logger.info(f"meta_epoch {meta_epoch}, hist_ind {hist_ind}, meta_eval_loss {meta_eval_loss:.4f},meta_eval_acc {meta_eval_accu:.4f}")
                self.logger.info(f"meta_eval_F1 {meta_eval_f1_score:.4f}, meta_eval_RECALL {meta_eval_recall:.4f}, meta_eval_PRECISION {meta_eval_precision:.4f}")
                self.logger.info(f"target meta_eval_{self.config.meta_target} {meta_eval_target:.4f}, meta_eval_prob_thresh {meta_thresh:.2f}")
                # if meta_eval_target > best_eval_target:
                #     best_eval_target = meta_eval_target
                #     self.best_meta_thresh = meta_thresh
                #     self.best_meta_model = deepcopy(self.meta_model)
                #     self.logger.info(f"better meta_model detected, saving..............")
                #     torch.save({'model_state_dict':self.meta_model.state_dict(),
                #         'optimizer_state_dict':self.meta_optimizer.state_dict(),'epoch':meta_epoch},
                #         self.ssl_path +f'/meta_checkpoint_best.pt')
                # self.meta_early_stopping(meta_eval_target,self.logger)
                if meta_eval_loss < best_eval_loss:
                    best_eval_loss = meta_eval_loss
                    self.best_meta_model = deepcopy(self.meta_model)
                    self.best_meta_thresh = meta_thresh
                    best_acc,best_f1,best_precistion,best_recall,best_target =  meta_eval_accu,meta_eval_f1_score,meta_eval_precision,meta_eval_recall,meta_eval_target
                    self.logger.info(f"better meta_model detected, saving..............")
                    torch.save({'model_state_dict':self.meta_model.state_dict(),
                        'optimizer_state_dict':self.meta_optimizer.state_dict(),'epoch':meta_epoch},
                        self.ssl_path +f'/meta_checkpoint_best.pt')
                self.meta_early_stopping(meta_eval_loss,self.logger)

                if self.meta_early_stopping.early_stop:
                    self.logger.info(f"meta model training early stop..............best meta_eval threshold {self.best_meta_thresh}")
                    self.meta_model = deepcopy(self.best_meta_model)
                    self.logger.info(f"best meta_eval ACC {best_acc}, F1 {best_f1}, PRECISION {best_precistion}, RECALL {best_recall}, optimize target {self.config.meta_target} {best_target}")
                    break
            feature_all = []
            for idx, feature_label in feature_batch.items():
                feature_all.append(feature_label)
            feature_all = torch.cat(feature_all, dim=0)
            cor_all = np.corrcoef(feature_all.cpu().t())
            feat_label_cor_all = cor_all[-1,:-1]
            # torch.save(feat_label_cor_all, self.ssl_path + f'/feature_label_all_cor_{epoch}_{hist_ind}.pt')
            # torch.save(feature_batch, self.ssl_path + f'/feature_batch_{epoch}_{hist_ind}.pt')
            # torch.save(cor_batch, self.ssl_path + f'/cor_batch_{epoch}_{hist_ind}.pt')
            torch.save(feat_label_cor_all, self.ssl_path + f'/feature_label_all_cor_{epoch}.pt')
            torch.save(feature_batch, self.ssl_path + f'/feature_batch_{epoch}.pt')
            torch.save(cor_batch, self.ssl_path + f'/cor_batch_{epoch}.pt')

    def infer_hist(self,check_signal=False):
        
        #load init model
        
        self.keys = ["Hist Losses","Instant Losses"]
        if self.config.use_entropy:
            self.keys.extend(["Hist Entropy","Instant Entropy"])
        if self.config.soft_entropy:
            self.keys.extend(['Soft Entropy'])
        if self.config.soft_entropy_mul:
            self.keys.extend(['Soft Entropy_Mul'])
        if self.config.use_simi:
            self.keys.append("First Last Similarity")
        #start inference for history loss,prediction
        with torch.no_grad():
            # for update_ind in range(self.config.hist_len):
            for update_ind in range(self.total_hist_len):
                #reload model
                checkpoint_path = self.sup_path +f'/checkpoint_{update_ind}.pt'
                self.load(checkpoint_path)
                self.logger.info(f"current inference on history {update_ind}")
                self.model.eval() #should this be in train mode? because we are recording history information
                #infer on unlabel set
                unlabel_hist_losses = []
                unlabel_hist_entropy = []
                unlabel_hist_entropy_soft = []
                unlabel_instant_losses = []
                unlabel_instant_entropy = []
                unlabel_simi = []
                unlabel_sifts = []
                unlabel_last_layer_hid = []
                unlabel_first_layer_hid = []
                unlabel_last_first_sim = []
                for idx, (src,targets,inds,ori_targets) in enumerate(self.unlabel_loader):
                    if self.debug and idx >= self.debug_sample_thres: 
                        break
                    if self.config.pretrained_model == "bert":
                        ids,attention_mask,token_type_ids = self.get_inputs(src)
                    else:
                        ids,attention_mask = self.get_inputs(src)
                    sift = targets == ori_targets
                    sift = sift.cpu().numpy()
                    targets = targets.to(self.device, dtype=torch.long)
                    if self.config.pretrained_model == "bert":
                        outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                    else:
                        outputs = self.model(ids, attention_mask, labels=targets)
                    loss, logits = outputs[0], outputs[1]
                    if self.config.use_simi:
                        hidden_states = outputs[2][1:] #first one is embedding
                        pads = torch.from_numpy(np.array(src['attention_mask'])).to(self.device)
                        simi = self.similarity_standard(hidden_states[0],hidden_states[-1])
                        simi = (simi * pads).sum(dim=-1) / pads.sum(dim=-1)
                        normalized_simi = self.min_max_normalize(simi)
                    loss_sample = self.cross_entropy_sample(logits, targets) # ori_targets?
                    normalized_loss = self.min_max_normalize(loss_sample)
                    pred = torch.argmax(logits, dim=-1)
                    conf_max, _ = torch.softmax(logits, dim=-1).max(dim=1)
                    conf = torch.softmax(logits, dim=-1)

                    if update_ind == 0:
                        previous_hist_loss = 0
                    else:
                        previous_hist_loss = self.unlabel_hist_loss[inds,update_ind-1]
                    self.unlabel_hist_loss[inds,update_ind] = self.config.loss_gamma * previous_hist_loss \
                                                    + (1 - self.config.loss_gamma) * normalized_loss.cpu().numpy()
                    self.unlabel_hist_track[inds,update_ind] = pred.cpu().numpy().astype(np.int32)
                    self.unlabel_hist_prob[inds,update_ind] = conf.cpu().numpy()
                    

                    # if update_ind == self.config.hist_len - 1:
                        # self.unlabel_instant_loss[inds] = normalized_loss.cpu().numpy()
                    self.unlabel_instant_loss[inds,update_ind] = normalized_loss.cpu().numpy()
                    if self.config.use_simi:
                        self.unlabel_instant_simi[inds,update_ind] = normalized_simi.cpu().numpy()
                    if check_signal:
                        if self.config.use_entropy:
                            cur_batch_hist_entropy,cur_batch_instant_entropy = self.calc_entropy(self.unlabel_hist_track[inds,:(update_ind+1)],type='hard')
                            cur_batch_hist_entropy,cur_batch_instant_entropy = cur_batch_hist_entropy.cpu().numpy(),cur_batch_instant_entropy.cpu().numpy()
                            unlabel_hist_entropy.append(cur_batch_hist_entropy)
                            unlabel_instant_entropy.append(cur_batch_instant_entropy)
                        if self.config.soft_entropy:
                            cur_batch_hist_entropy_soft,_ = self.calc_entropy(self.unlabel_hist_prob[inds,:(update_ind+1)],type='soft')
                            unlabel_hist_entropy_soft.append(cur_batch_hist_entropy_soft.cpu().numpy())

                        if self.config.use_simi:
                            cur_batch_simi = normalized_simi.cpu().numpy()
                            unlabel_simi.append(cur_batch_simi)
                        cur_batch_hist_losses,cur_batch_instant_losses = self.unlabel_hist_loss[inds,update_ind],normalized_loss.cpu().numpy()
                        unlabel_hist_losses.append(cur_batch_hist_losses)
                        unlabel_instant_losses.append(cur_batch_instant_losses)
                        unlabel_sifts.append(sift)
                #mind the order,should be same as keys = ["Hist Losses","Instant Losses","Hist Entropy","Instant Entropy"]
                if check_signal:
                    unlabel_signals = [unlabel_hist_losses,unlabel_instant_losses]
                    if self.config.use_entropy:
                        unlabel_signals.extend([unlabel_hist_entropy,unlabel_instant_entropy])
                    if self.config.soft_entropy:
                        unlabel_signals.extend([unlabel_hist_entropy_soft])
                    if self.config.use_simi:
                        unlabel_signals.append(unlabel_simi)
                    unlabel_analysis_res = self.check_signal(unlabel_signals,unlabel_sifts)
                
                #infer on meta set
                meta_hist_losses = []
                meta_hist_entropy = []
                meta_hist_entropy_soft = []
                meta_instant_losses = []
                meta_instant_entropy = []
                meta_last_layer_hid = []
                meta_first_layer_hid = []
                meta_last_first_sim = []
                meta_sifts = []
                for _, (src,targets,inds,ori_targets) in enumerate(self.meta_loader):
                    if self.config.pretrained_model == "bert":
                        ids,attention_mask,token_type_ids = self.get_inputs(src)
                    else:
                        ids,attention_mask = self.get_inputs(src)
                    sift = targets == ori_targets
                    sift = sift.cpu().numpy()
                    targets = targets.to(self.device, dtype=torch.long)
                    if self.config.pretrained_model == "bert":
                        outputs = self.model(ids, attention_mask, token_type_ids, labels=targets) 
                    else:
                        outputs = self.model(ids, attention_mask, labels=targets) 
                    loss, logits = outputs[0], outputs[1]
                    if self.config.use_simi:
                        hidden_states = outputs[2][1:]
                        pads = torch.from_numpy(np.array(src['attention_mask'])).to(self.device)
                        simi = self.similarity_standard(hidden_states[0],hidden_states[-1])
                        simi = (simi * pads).sum(dim=-1) / pads.sum(dim=-1)
                        normalized_simi = self.min_max_normalize(simi)
                    loss_sample = self.cross_entropy_sample(logits, targets)
                    normalized_loss = self.min_max_normalize(loss_sample)
                    pred = torch.argmax(logits, dim=-1)
                    conf_max, _ = torch.softmax(logits, dim=-1).max(dim=1)
                    conf = torch.softmax(logits, dim=-1)
                    if update_ind == 0:
                        previous_hist_loss = 0
                    else:
                        previous_hist_loss = self.meta_hist_loss[inds,update_ind-1]
                    self.meta_hist_loss[inds,update_ind] = self.config.loss_gamma * previous_hist_loss \
                                                    + (1 - self.config.loss_gamma) * normalized_loss.cpu().numpy()
                    self.meta_hist_track[inds,update_ind] = pred.cpu().numpy()
                    self.meta_hist_prob[inds,update_ind] = conf.cpu().numpy()
                    # if update_ind == self.config.hist_len - 1:
                        # self.meta_instant_loss[inds] = normalized_loss.cpu().numpy()
                    self.meta_instant_loss[inds,update_ind] = normalized_loss.cpu().numpy()
                    if self.config.use_simi:
                        self.meta_instant_simi[inds,update_ind] = normalized_simi.cpu().numpy()
                    if check_signal:
                        if self.config.use_entropy:
                            cur_batch_hist_entropy,cur_batch_instant_entropy = self.calc_entropy(self.meta_hist_track[inds,:(update_ind+1)],type='hard')
                            cur_batch_hist_entropy,cur_batch_instant_entropy = cur_batch_hist_entropy.cpu().numpy(),cur_batch_instant_entropy.cpu().numpy()
                            meta_hist_entropy.append(cur_batch_hist_entropy)
                            meta_instant_entropy.append(cur_batch_instant_entropy)
                        if self.config.soft_entropy:
                            cur_batch_hist_entropy_soft = self.calc_entropy(self.meta_hist_prob[inds,:(update_ind+1)],type='soft')
                            meta_hist_entropy_soft.append(cur_batch_hist_entropy_soft.cpu().numpy())
                        if self.config.use_simi:
                            cur_batch_simi = normalized_simi.cpu().numpy()
                            meta_last_first_sim.append(cur_batch_simi)
                        cur_batch_hist_losses,cur_batch_instant_losses = self.meta_hist_loss[inds,update_ind],normalized_loss.cpu().numpy()
                        meta_hist_losses.append(cur_batch_hist_losses)
                        meta_instant_losses.append(cur_batch_instant_losses)
                        meta_sifts.append(sift)
                if check_signal:
                    meta_signals = [meta_hist_losses,meta_instant_losses]
                    if self.config.use_entropy:
                        meta_signals.extend([meta_hist_entropy,meta_instant_entropy])
                    if self.config.soft_entropy:
                        meta_signals.extend([meta_hist_entropy_soft])
                    if self.config.use_simi:
                        meta_signals.append(meta_last_first_sim)
                    meta_analysis_res = self.check_signal(meta_signals,meta_sifts)
                #infer on dev set, for evaluation during meta model training
                dev_hist_losses = []
                dev_hist_entropy = []
                dev_hist_entropy_soft = []
                dev_instant_losses = []
                dev_instant_entropy = []
                dev_last_layer_hid = []
                dev_first_layer_hid = []
                dev_last_first_sim = []
                dev_sifts = []

                for _, (src,targets,inds,ori_targets) in enumerate(self.valid_loader):
                    if self.config.pretrained_model == "bert":
                        ids,attention_mask,token_type_ids = self.get_inputs(src)
                    else:
                        ids,attention_mask = self.get_inputs(src)
                    sift = targets == ori_targets
                    sift = sift.cpu().numpy()
                    targets = targets.to(self.device, dtype=torch.long)
                    if self.config.pretrained_model == "bert":
                        outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                    else:
                        outputs = self.model(ids, attention_mask, labels=targets)
                    loss, logits = outputs[0], outputs[1] # opt
                    if self.config.use_simi:
                        hidden_states = outputs[2][1:] #first one is embedding
                        pads = torch.from_numpy(np.array(src['attention_mask'])).to(self.device)
                        simi = self.similarity_standard(hidden_states[0],hidden_states[-1])
                        simi = (simi * pads).sum(dim=-1) / pads.sum(dim=-1)
                        normalized_simi = self.min_max_normalize(simi)
                    loss_sample = self.cross_entropy_sample(logits, targets)
                    normalized_loss = self.min_max_normalize(loss_sample)
                    pred = torch.argmax(logits, dim=-1)
                    conf_max, _ = torch.softmax(logits, dim=-1).max(dim=1)
                    conf = torch.softmax(logits, dim=-1)
                    if update_ind == 0:
                        previous_hist_loss = 0
                    else:
                        previous_hist_loss = self.dev_hist_loss[inds,update_ind-1]
                    self.dev_hist_loss[inds,update_ind] = self.config.loss_gamma * previous_hist_loss \
                                                    + (1 - self.config.loss_gamma) * normalized_loss.cpu().numpy()
                    self.dev_hist_track[inds,update_ind] = pred.cpu().numpy()
                    self.dev_hist_prob[inds,update_ind] = conf.cpu().numpy()
                    # if update_ind == self.config.hist_len - 1:
                    self.dev_instant_loss[inds,update_ind] = normalized_loss.cpu().numpy()
                    if self.config.use_simi:
                        self.dev_instant_simi[inds,update_ind] = normalized_simi.cpu().numpy()
                    if check_signal:
                        if self.config.use_entropy:
                            cur_batch_hist_entropy,cur_batch_instant_entropy = self.calc_entropy(self.dev_hist_track[inds,:(update_ind+1)],type='hard')
                            cur_batch_hist_entropy,cur_batch_instant_entropy = cur_batch_hist_entropy.cpu().numpy(),cur_batch_instant_entropy.cpu().numpy()
                            dev_hist_entropy.append(cur_batch_hist_entropy)
                            dev_instant_entropy.append(cur_batch_instant_entropy)
                        if self.config.soft_entropy:
                            cur_batch_hist_entropy_soft = self.calc_entropy(self.dev_hist_prob[inds,:(update_ind+1)],type='soft')
                            dev_hist_entropy_soft.append(cur_batch_hist_entropy_soft.cpu().numpy())
                        if self.config.use_simi:
                            cur_batch_simi = normalized_simi.cpu().numpy()
                            dev_last_first_sim.append(cur_batch_simi)
                        cur_batch_hist_losses,cur_batch_instant_losses = self.dev_hist_loss[inds,update_ind],normalized_loss.cpu().numpy()
                        dev_hist_losses.append(cur_batch_hist_losses)
                        dev_instant_losses.append(cur_batch_instant_losses)
                        dev_sifts.append(sift)
                if check_signal:
                    dev_signals = [dev_hist_losses,dev_instant_losses]
                    if self.config.use_entropy:
                        dev_signals.extend([dev_hist_entropy,dev_instant_entropy])
                    if self.config.soft_entropy:
                        dev_signals.extend([dev_hist_entropy_soft])
                    if self.config.use_simi:
                        dev_signals.append(dev_last_first_sim)
                    dev_analysis_res = self.check_signal(dev_signals,dev_sifts)
                    results = {'unlabel':unlabel_analysis_res,'meta_train':meta_analysis_res,'meta_dev':dev_analysis_res}
                    for loads,res in results.items():
                        self.logger.info(f"################ {loads} signals results ################")
                        for i,key in enumerate(self.keys):
                            crct_res,incrct_res = res["correct"][i],res["incorrect"][i]
                            try:
                                crct_percentiles = np.percentile(crct_res,[25,50,75])
                            except:
                                crct_percentiles = "No correct"
                            try:
                                incrct_percentiles = np.percentile(incrct_res,[25,50,75])
                            except:
                                incrct_percentiles = "No incorrect"
                            self.logger.info(f"correct {key} percentiles {crct_percentiles}, incorrect {key} percentiles {incrct_percentiles}")
                
    def cross_entropy_sample(self, logits, labels):
        num_class = self.config.class_num
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, num_class), labels.view(-1))
        return loss

    def pseudo_labeling(self,dataloader, guide_type=None):
        self.model.eval()
        # new_dataset = {label:[] for label in range(self.config.class_num)}
        pseudo_labeling_chancge_num = 0
        total = 0
        origin_target_crct_num = 0
        current_crct_num = 0
        new_total = 0
        after_change_correct_verify = 0
        with torch.no_grad():
            for idx,(src,targets,inds,ori_targets) in enumerate(dataloader):
                if self.debug and idx >= self.debug_sample_thres:
                    break
                total += len(targets)
                if self.config.pretrained_model == "bert":
                    ids,attention_mask,token_type_ids = self.get_inputs(src)
                else:
                    ids,attention_mask = self.get_inputs(src)
                targets = targets.to(self.device, dtype=torch.long)
                ori_targets = ori_targets.to(self.device, dtype=torch.long)
                # outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                if self.config.pretrained_model == "bert":
                    outputs = self.global_best_model(ids, attention_mask, token_type_ids, labels=targets)
                else:
                    outputs = self.global_best_model(ids, attention_mask, labels=targets)
                loss, logits = outputs[0], outputs[1]
                confidences = torch.softmax(logits, dim=-1)
                big_val, big_idx = torch.max(confidences.data, dim=-1)
                chancge_num = targets != big_idx
                pseudo_labeling_chancge_num += chancge_num.sum().item()
                origin_target_crct_num += (targets == ori_targets).sum().item()
                current_crct_num += (big_idx == ori_targets).sum().item()
                if guide_type is None:
                    dataloader.dataset._update_target(inds,big_idx.cpu().tolist())
        # to verify that we have change the loader
        # with torch.no_grad():
        #     for idx,(src,targets,inds,ori_targets) in enumerate(dataloader):
        #         if self.debug and idx >= self.debug_sample_thres:
        #             break
        #         new_total += len(targets)
        #         after_change_correct_verify += (targets == ori_targets).sum().item()
        self.logger.info(f"total samples {total}, pseudo_labeling_chancge_num {pseudo_labeling_chancge_num}, origin_target_crct_num {origin_target_crct_num}, current_crct_num {current_crct_num}")
        # self.logger.info(f"total samples {new_total}, after_change_correct_verify {after_change_correct_verify}")
        self.logger.info(f"original accuracy {(origin_target_crct_num*100 / total):.4f}, current accuracy {(current_crct_num*100 / total):.4f}")
        

    def select(self,threshold=0.5,type=None,epoch='None'):
        #train meta dataset
        self.model.eval()
        self.meta_model.eval()
        #infer on meta set
        select_data = {}
        meta_preds,meta_gt,meta_probs = [],[],[]

        with torch.no_grad():
            if not type and self.config.use_entropy:
                hist_entropy,cur_entropy = self.calc_entropy(self.unlabel_hist_track,type='hard')
                hist_entropy,cur_entropy = hist_entropy.to(self.device),cur_entropy.to(self.device)
            if not type and self.config.soft_entropy:
                if self.config.soft_entropy_mul:
                    hist_entropy_soft, hist_entropy_soft_mul = self.calc_entropy(self.unlabel_hist_prob,type='soft')
                    hist_entropy_soft_mul[torch.isinf(hist_entropy_soft_mul)] = 0
                    hist_entropy_soft_mul = hist_entropy_soft_mul.to(self.device)
                else:
                    hist_entropy_soft, _ = self.calc_entropy(self.unlabel_hist_prob,type='soft')
                hist_entropy_soft[torch.isinf(hist_entropy_soft)] = 0
                hist_entropy_soft = hist_entropy_soft.to(self.device)
            if not type:
                hist_loss_all,cur_loss_all = torch.from_numpy(self.unlabel_hist_loss[:,self.total_hist_len-1]).to(self.device),torch.from_numpy(self.unlabel_instant_loss[:,self.total_hist_len-1]).to(self.device)
                if self.config.use_simi:
                    cur_simi_all = torch.from_numpy(self.unlabel_instant_simi[:,self.total_hist_len-1]).to(self.device)
            n_correct, n_total, n_select,n_max_positive = 0, 0, 0, 0
            changed_num,crct_num =0,0
            meta_unlabel_loss = 0
            feature_batch = {}
            cor_batch = {}
            for idx, (src,targets,inds,ori_targets) in enumerate(self.unlabel_loader):
                # tmp_targets = targets
                if not type:
                    # tmp_targets = torch.from_numpy(self.unlabel_hist_track[:,self.total_hist_len-1][inds]).long()
                    tmp_targets = targets
                    changed_num += (tmp_targets != ori_targets).sum().item()
                    crct_num += (tmp_targets == ori_targets).sum().item()
                targets = targets.to(self.device)
                ori_targets = ori_targets.to(self.device, dtype=torch.long)
                training_targets = targets == ori_targets
                n_max_positive += training_targets.sum()
                meta_gt.extend(training_targets.long().cpu().tolist())
                if not type:
                    hist_loss,cur_loss = hist_loss_all[inds],cur_loss_all[inds]
                    if self.config.use_simi:
                        cur_simi = cur_simi_all[inds]
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
                    if self.config.soft_entropy:
                        input_signals.extend([hist_entropy_soft[inds][:,None]])
                    if self.config.soft_entropy_mul:
                        for i in range(self.config.class_num):
                            hist_entropy_soft_mul_i = hist_entropy_soft_mul[:,i]
                            input_signals.extend([hist_entropy_soft_mul_i[inds][:,None]])
                    if self.config.use_simi:
                        input_signals.append(cur_simi[:,None] if len(cur_simi.shape) == 1 else cur_simi)
                    
                    select_label = training_targets.long()
                    feature = torch.cat(input_signals,dim=-1).float()
                    feature_label = torch.cat([feature,select_label.reshape(-1,1)],dim=-1)
                    cor = np.corrcoef(feature_label.cpu().t())
                    feat_label_cor = cor[-1,:-1]
                    feature_batch[idx] = feature_label
                    cor_batch[idx] = feat_label_cor
                    
                    if self.config.test_single_feat:
                        feat_name_dict = {"hist_emaloss":0,"cur_loss":1,"hist_entropy":2,"cur_entropy":3,"soft_entropy":4,"cur_simi":5}
                        input_feature = input_signals[feat_name_dict[self.config.test_single_feat_name]]
                        if self.config.test_top_feat:
                            top_rate = self.config.top_rate
                            if self.config.positive:
                                thres = np.percentile(input_feature.cpu(), 100*(1-top_rate))
                            else:   
                                thres = np.percentile(input_feature.cpu(), 100*top_rate)
                        meta_logits = self.meta_model(input_feature.float())
                    else:
                        meta_logits = self.meta_model(torch.cat(input_signals,dim=-1).float())

                    meta_prob = F.softmax(meta_logits,dim=-1)
                    meta_probs.extend(meta_prob[:,1].cpu().tolist())
                    meta_loss = self.meta_ce_loss(meta_logits,training_targets.long())
                    # pred = torch.argmax(meta_logits, dim=-1)
                    # meta_preds.extend(pred.cpu().tolist())
                    if self.config.test_top_feat:
                        if self.config.positive:
                            sift_pos = (input_feature>=thres).squeeze()
                            # inds_list = inds[sift_pos].cpu().squeeze().tolist()
                        else:
                            sift_pos = (input_feature<=thres).squeeze()
                            # inds_list = inds[sift_pos].cpu().squeeze().tolist()
                    else:
                        sift_pos = meta_prob[:,1] >= self.best_meta_thresh
                    pred = sift_pos.int()
                    meta_preds.extend(pred.cpu().tolist())
                    inds_list = inds[pred == 1].cpu().tolist()
                    selected_num = pred.sum().item()
                else:
                    if self.config.pretrained_model == "bert":
                        ids,attention_mask,token_type_ids = self.get_inputs(src)
                        outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                    else:
                        ids,attention_mask = self.get_inputs(src)
                        outputs = self.model(ids, attention_mask, labels=targets)
                    meta_loss, logits = outputs[0], outputs[1]
                    bsz = logits.shape[0]
                    meta_prob = F.softmax(logits,dim=-1)
                    argmax_ids = torch.argmax(meta_prob,dim=-1)
                    sift_pos = meta_prob[torch.arange(bsz),argmax_ids] >= threshold
                    inds_list = inds[sift_pos].cpu().tolist()
                    selected_num = sift_pos.sum().item()
                    meta_preds.extend(sift_pos.long().cpu().tolist())
                    
                n_total += targets.size(0)
                # n_select += selected_num
                for i,ind in enumerate(inds_list):
                    select_data[ind] = deepcopy(self.unlabel_loader.dataset._data[ind])
                    n_select += 1
                meta_unlabel_loss += meta_loss.item()
            n_correct = (np.array(meta_gt) == np.array(meta_preds)).sum()
            acc = n_correct/n_total
            precision, recall, f1_score, support = precision_recall_fscore_support(meta_gt, meta_preds,average='binary')
            beta = 0.0625
            fx_score = (1+beta**2)*(precision*recall) / (beta**2*precision+recall)
            if not type:
                optimize_target,unlabel_threshold,_, _, _,_,_,_ = opt_threshold(meta_gt,meta_probs,self.config.meta_target,self.config.beta)
            self.logger.info(f"unlabel changed_num is {changed_num}, unlabel correct inferred {crct_num}, unlabel acc inferred {(crct_num / n_total):.4f}")
            self.logger.info(f"meta_unlabel_loss {meta_unlabel_loss/ (idx+1):.4f},meta_unlabel_acc {acc:.4f}, # selected samples {n_select}, # total samples {n_total}, #max positive samples {n_max_positive}")
            self.logger.info(f"meta_unlabel_F1 {f1_score:.4f}, meta_unlabel_RECALL {recall:.4f}, meta_unlabel_PRECISION {precision:.4f}, meta_unlabel_FX {fx_score:.4f}, beta {beta}")
            if not type:
                self.logger.info(f"meta_unlabel_best_possible_{self.config.meta_target} {optimize_target:.4f}, meta_unlabel_best_possible_THRESHOLD {unlabel_threshold:.2f}, current threshold {self.best_meta_thresh:.2f}")
            if not type:
                feature_all = []
                for idx, feature_label in feature_batch.items():
                    feature_all.append(feature_label)
                feature_all = torch.cat(feature_all, dim=0)
                cor_all = np.corrcoef(feature_all.cpu().t())
                feat_label_cor_all = cor_all[-1,:-1]
                torch.save(feat_label_cor_all, self.ssl_path + f'/unlabel_feature_label_all_cor_{epoch}.pt')
                torch.save(feature_batch, self.ssl_path + f'/unlabel_feature_batch_{epoch}.pt')
                torch.save(cor_batch, self.ssl_path + f'/unlabel_cor_batch_{epoch}.pt')
        return select_data

    def build_lexicon(self, input_ids, targets, attns):
        top_k = 3 
        values, indices = torch.topk(attns, top_k, dim=-1)
        decoded_inputs = self.tokenizer.batch_decode(input_ids)
        
        for input_id, sent, seq_idxs, attn, label in zip(input_ids, decoded_inputs, indices, attns, targets):
            words = self.tokenizer.tokenize(sent)
            cleaned_words = self.tokenizer.decode(input_id, skip_special_tokens=True)
            label = label.item()
            
            if len(self.tokenizer.tokenize(cleaned_words)) <= top_k:
                # choose top one
                vocab_id = input_id[seq_idxs[0].item()].item()
                word = self.tokenizer.convert_ids_to_tokens(vocab_id)
                if '#' in word or len(word) <=2 or word in stop_words:
                    continue
                word = self.lemmatizer.lemmatize(word)
                if word in self.lexicon_temp[label]:
                    self.lexicon_temp[label][word] +=1
                else:
                    self.lexicon_temp[label][word] = 1
            else:
                # choose top three
                vocab_ids = [input_id[idx.item()].item() for idx in seq_idxs]
                words = self.tokenizer.convert_ids_to_tokens(vocab_ids)
                for word in words:
                    if '#' in word or len(word) <=2 or word in stop_words:
                        continue
                    word = self.lemmatizer.lemmatize(word)
                    if word in self.lexicon_temp[label]:
                        self.lexicon_temp[label][word] += 1
                    else:
                        self.lexicon_temp[label][word] = 1
    
    
    def add_dataset(self, labeled_dataset, new_dataset=None, add_meta_ratio = 0,meta_dataset = None):
        #generat a deepcopy of label dataset, and update that copy, do not change original dataset
        if new_dataset is None:
            return labeled_dataset,None
        res_data = deepcopy(labeled_dataset._data)
        leng = len(labeled_dataset)
        add_meta_num = 0
        if add_meta_ratio > 0:
            add_meta_num = max(1,int(add_meta_ratio) * len(new_dataset)) #incase zero meta num
            meta_data = deepcopy(meta_dataset._data)
            leng_meta = len(meta_dataset)
        pop_list = []
        for i,(k,v) in enumerate(new_dataset.items()): # add meta
            if add_meta_num == 0:
                break
            else:
                add_meta_num -= 1
                v['index'] = leng_meta+i
                meta_data[leng_meta+i] = deepcopy(v)
                pop_list.append(k)
        for k in pop_list:
            new_dataset.pop(k)
        for i,(k,v) in enumerate(new_dataset.items()):
            assert res_data.get(leng+i) is None
            v['index'] = leng+i
            res_data[leng+i] = deepcopy(v)

        if add_meta_num > 0:
            return DatasetXL(res_data,self.label_mapping),DatasetXL(meta_data,self.label_mapping)
        else:
            return DatasetXL(res_data,self.label_mapping),None
    
    
    def remove_dataset(self, unlabeled_dataset, new_dataset):
        unlabeled_texts = [data[0] for data in unlabeled_dataset]
        unlabeled_labels = [data[1] for data in unlabeled_dataset]
        
        new_texts = [data[0] for data in new_dataset]
        new_labels = [data[1] for data in new_dataset]
        
        # remove pseudo-labeled from unlabeled dataset
        for text in new_texts:
            idx = unlabeled_texts.index(text)
            unlabeled_texts.pop(idx)
            unlabeled_labels.pop(idx)
                    
        return list(zip(unlabeled_texts, unlabeled_labels))
    
        
    def encode_dataset(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        dataset = Dataset(encodings, labels)
        return dataset
    
    
    def decode_dataset(self, dataset):
        decoded_texts = []
        labels = []
        for idx in range(len(dataset)):
            text_id = dataset[idx]['input_ids']
            label = dataset[idx]['labels'].item()
            decoded_text = self.tokenizer.decode(text_id, skip_special_tokens=True)
            decoded_texts.append(decoded_text)
            labels.append(label)
        return decoded_texts, labels

    def min_max_normalize(self,target):
        min_val = torch.min(target)
        max_val = torch.max(target)
        normalized_target = (target - min_val.detach()) / ((max_val - min_val) + torch.finfo(target.dtype).eps).detach()
        return normalized_target
    
    def get_inputs(self,src):
        ids = torch.tensor(src['input_ids']).to(self.device, dtype=torch.long)
        attention_mask = torch.tensor(src['attention_mask']).to(self.device, dtype=torch.long)
        if self.config.pretrained_model == "bert":
            token_type_ids = torch.tensor(src['token_type_ids']).to(self.device, dtype=torch.long)
            return ids,attention_mask,token_type_ids
        return ids,attention_mask
    
    def calc_entropy(self,pred_history, type="hard"):
        if type == "soft":
            if len(pred_history.shape) == 1:
                pred_history = pred_history.reshape(1,-1)
            nums = len(pred_history) # batch size
            hist_uncer_np = np.zeros(shape=(nums,))
            hist_entro_mul = np.zeros(shape=(nums,self.config.class_num))
            if pred_history.shape[1] <= 2:
                hist_uncer_tc = torch.from_numpy(hist_uncer_np).to(self.device)
                hist_entro_mul_tc = torch.from_numpy(hist_entro_mul).to(self.device)
                return hist_uncer_tc, hist_entro_mul_tc
            else:
                for i,pred_labels_for_one_sample in enumerate(pred_history):
                    value = continuous.get_h(pred_labels_for_one_sample, k=2)
                    value_gaussian = continuous.get_h_mvn(pred_labels_for_one_sample)
                    value_class = np.zeros(shape=(self.config.class_num,))
                    for j in range(self.config.class_num):
                        value_class[j] = continuous.get_h_mvn(pred_labels_for_one_sample[:,j])
                    hist_uncer_np[i] = value
                    hist_entro_mul[i] = value_class
                hist_uncer_tc = torch.from_numpy(hist_uncer_np).to(self.device)
                hist_entro_mul_tc = torch.from_numpy(hist_entro_mul).to(self.device)
                hist_entro_mul_norm = torch.zeros(size=(nums,self.config.class_num))
                for s in range(nums):
                    hist_entro_mul_norm[s] = self.min_max_normalize(hist_entro_mul_tc[s,:])
                return self.min_max_normalize(hist_uncer_tc), hist_entro_mul_norm
        if type == "hard":
            if len(pred_history.shape) == 1:
                pred_history = pred_history.reshape(1,-1)
            nums = len(pred_history)
            hist_uncer_np = np.zeros(shape=(nums,))
            instant_uncer_np = np.zeros(shape=(nums,))
            for i,pred_labels_for_one_sample in enumerate(pred_history):
                c = Counter(pred_labels_for_one_sample)
                keys = list(c.keys())
                values = np.array(list(c.values()))
                values = values / (self.config.hist_len)
                values = values * np.log(values)
                values = -1 * values / self.max_uncertainty
                pred_key = pred_labels_for_one_sample[-1]
                if not pred_key in keys:
                    instant_uncer_np[i] = 0
                else:
                    index = keys.index(pred_key)
                    instant_uncer_np[i] = values[index]
                uncertainty = values.sum()
                hist_uncer_np[i] = uncertainty
            hist_uncer_tc = torch.from_numpy(hist_uncer_np).to(self.device)
            instant_uncer_tc = torch.from_numpy(instant_uncer_np).to(self.device)
            return self.min_max_normalize(hist_uncer_tc),self.min_max_normalize(instant_uncer_tc)
    def load(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def load_hist(self,outer_epoch):
        self.unlabel_hist_track = np.load(os.path.join(self.ssl_path,f"unlabel_hist_track_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.unlabel_hist_prob = np.load(os.path.join(self.ssl_path,f"unlabel_hist_prob_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.unlabel_hist_loss = np.load(os.path.join(self.ssl_path,f"unlabel_hist_loss_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.unlabel_instant_loss = np.load(os.path.join(self.ssl_path,f"unlabel_instant_loss_{outer_epoch}_epoch.npy"),allow_pickle=True)

        self.meta_hist_track = np.load(os.path.join(self.ssl_path,f"meta_hist_track_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.meta_hist_prob = np.load(os.path.join(self.ssl_path,f"meta_hist_prob_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.meta_hist_loss = np.load(os.path.join(self.ssl_path,f"meta_hist_loss_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.meta_instant_loss = np.load(os.path.join(self.ssl_path,f"meta_instant_loss_{outer_epoch}_epoch.npy"),allow_pickle=True)

        self.dev_hist_track = np.load(os.path.join(self.ssl_path,f"dev_hist_track_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.dev_hist_prob = np.load(os.path.join(self.ssl_path,f"dev_hist_prob_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.dev_hist_loss = np.load(os.path.join(self.ssl_path,f"dev_hist_loss_{outer_epoch}_epoch.npy"),allow_pickle=True)
        self.dev_instant_loss = np.load(os.path.join(self.ssl_path,f"dev_instant_loss_{outer_epoch}_epoch.npy"),allow_pickle=True)

    def save_hist(self,outer_epoch):
        np.save(os.path.join(self.ssl_path,f"unlabel_hist_track_{outer_epoch}_epoch"),self.unlabel_hist_track)
        np.save(os.path.join(self.ssl_path,f"unlabel_hist_prob_{outer_epoch}_epoch"),self.unlabel_hist_prob)
        np.save(os.path.join(self.ssl_path,f"unlabel_hist_loss_{outer_epoch}_epoch"),self.unlabel_hist_loss)
        np.save(os.path.join(self.ssl_path,f"unlabel_instant_loss_{outer_epoch}_epoch"),self.unlabel_instant_loss)

        np.save(os.path.join(self.ssl_path,f"meta_hist_track_{outer_epoch}_epoch"),self.meta_hist_track)
        np.save(os.path.join(self.ssl_path,f"meta_hist_prob_{outer_epoch}_epoch"),self.meta_hist_prob)
        np.save(os.path.join(self.ssl_path,f"meta_hist_loss_{outer_epoch}_epoch"),self.meta_hist_loss)
        np.save(os.path.join(self.ssl_path,f"meta_instant_loss_{outer_epoch}_epoch"),self.meta_instant_loss)

        np.save(os.path.join(self.ssl_path,f"dev_hist_track_{outer_epoch}_epoch"),self.dev_hist_track)
        np.save(os.path.join(self.ssl_path,f"dev_hist_prob_{outer_epoch}_epoch"),self.dev_hist_prob)
        np.save(os.path.join(self.ssl_path,f"dev_hist_loss_{outer_epoch}_epoch"),self.dev_hist_loss)
        np.save(os.path.join(self.ssl_path,f"dev_instant_loss_{outer_epoch}_epoch"),self.dev_instant_loss)
    



