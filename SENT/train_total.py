import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_set', required = False,default="imdb")
parser.add_argument('--train_path', required = False,default="/home/self-training/aclImdb/train")
parser.add_argument('--train_set', required = False,default="small")
parser.add_argument('--test_path', required = False,default = "/home/self-training/aclImdb/test")
parser.add_argument('--save_path', required = False,default ="./output")
parser.add_argument('--quick_train',type=bool,required=False,default = False)
parser.add_argument('--model_type', required=True, help='type among [ours, baseline, bert_attn]',default = 'baseline')
parser.add_argument('--do_augment', type=bool, default=False, required = False)
parser.add_argument('--num', type=int, default=0, required = False)
parser.add_argument('--test', type=bool, default=False, required = False)
parser.add_argument('--test_confidence', type=bool, default=False, required = False)
parser.add_argument('--test_drop_train', type=bool, default=False, required = False)
parser.add_argument('--num_drop', type=int, default=0, required = False)

parser.add_argument('--test_few', type=bool, default=False, required = False)
parser.add_argument('--num_few', type=int, default=0, required = False)
parser.add_argument('--gpu_id', type=int, default=0, required = False)

args = parser.parse_args()
import os
gpuid = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"

import pickle
import torch
import logging
import math
from util.datasetxl import *
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertModel
from model import BERT_ATTN
from torch.utils.data import DataLoader
from trainer_new import Trainer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
# from util.augment import *
import copy
import time
import warnings
import random 
import numpy as np
import json
import datetime
from collections import Counter
timestr= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(timestr)
warnings.filterwarnings('ignore')





args.save_path = os.path.join("/home/self-training/ours",f"output_{args.data_set}")

##########
# logger
##########

work_dir = args.save_path
model_type = args.model_type
data_set = args.data_set
num = args.num
args.save_path = os.path.join(args.save_path,f'{num}_{model_type}_{timestr}')
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
print(args.data_set)
print(args.save_path)
log_path = os.path.join(args.save_path, f'log.txt')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("#"*100)
logger.info(args.data_set)
logger.info("#"*100)

if args.data_set == "sms":
    train_dir = "/home/self-training/ours/data/sms/train_data.txt"
    dev_dir = "/home/self-training/ours/data/sms/dev_data.txt"
    meta_dev_dir = '/home/self-training/ours/data/sms/meta_dev_data.txt'
    test_dir = '/home/self-training/ours/data/sms/test_data.txt'
    unlabel_dir = "/home/self-training/ours/data/sms/unlabeled_data.txt"
    from constant_sms import Config
    config = Config()
elif args.data_set == 'trec':
    train_dir = "/home/self-training/ours/data/trec/train_data.txt"
    dev_dir = "/home/self-training/ours/data/trec/dev_data.txt"
    meta_dev_dir = '/home/self-training/ours/data/trec/meta_dev_data.txt'
    test_dir = '/home/self-training/ours/data/trec/test_data.txt'
    unlabel_dir = "/home/self-training/ours/data/trec/unlabeled_data.txt"
    from constant_trec import Config
    config = Config()
elif args.data_set == 'youtube':
    train_dir = "/home/self-training/ours/data/youtube/train_data.txt"
    dev_dir = "/home/self-training/ours/data/youtube/dev_data.txt"
    meta_dev_dir = '/home/self-training/ours/data/youtube/meta_dev_data.txt'
    test_dir = '/home/self-training/ours/data/youtube/test_data.txt'
    unlabel_dir = "/home/self-training/ours/data/youtube/unlabeled_data.txt"
    from constant_youtube import Config
    config = Config()
elif args.data_set == 'youtube_2':
    train_dir = "/home/self-training/ours/data/youtube/train_data_2.txt"
    dev_dir = "/home/self-training/ours/data/youtube/dev_data_2.txt"
    meta_dev_dir = '/home/self-training/ours/data/youtube/meta_dev_data_2.txt'
    test_dir = '/home/self-training/ours/data/youtube/test_data_2.txt'
    unlabel_dir = "/home/self-training/ours/data/youtube/unlabeled_data_2.txt"
    from constant_youtube import Config
    config = Config()
elif args.data_set == 'youtube_5':
    train_dir = "/home/self-training/ours/data/youtube/train_data_5.txt"
    dev_dir = "/home/self-training/ours/data/youtube/dev_data_5.txt"
    meta_dev_dir = '/home/self-training/ours/data/youtube/meta_dev_data_5.txt'
    test_dir = '/home/self-training/ours/data/youtube/test_data_5.txt'
    unlabel_dir = "/home/self-training/ours/data/youtube/unlabeled_data_5.txt"
    from constant_youtube import Config
    config = Config()
elif args.data_set == 'youtube_20':
    train_dir = "/home/self-training/ours/data/youtube/train_data_20.txt"
    dev_dir = "/home/self-training/ours/data/youtube/dev_data_20.txt"
    meta_dev_dir = '/home/self-training/ours/data/youtube/meta_dev_data_20.txt'
    test_dir = '/home/self-training/ours/data/youtube/test_data_20.txt'
    unlabel_dir = "/home/self-training/ours/data/youtube/unlabeled_data_20.txt"
    from constant_youtube import Config
    config = Config()
elif args.data_set == 'ag_news': # _2 is smaller one
    train_dir = "/home/self-training/ours/data/ag_news/train_data_2.txt"
    dev_dir = "/home/self-training/ours/data/ag_news/dev_data_2.txt"
    meta_dev_dir = '/home/self-training/ours/data/ag_news/meta_dev_data_2.txt'
    test_dir = '/home/self-training/ours/data/ag_news/test_data_2.txt'
    unlabel_dir = "/home/self-training/ours/data/ag_news/unlabeled_data_2.txt"
    from constant_ag import Config
    config = Config()
elif args.data_set == 'imdb':
    train_dir = "/home/self-training/ours/data/imdb50/train_data_shuf.txt"
    dev_dir = "/home/self-training/ours/data/imdb50/dev_data_shuf.txt"
    meta_dev_dir = "/home/self-training/ours/data/imdb50/meta_dev_data_shuf.txt"
    test_dir = "/home/self-training/ours/data/imdb50/test_data_shuf.txt"
    unlabel_dir = "/home/self-training/ours/data/imdb50/unlabeled_data_shuf.txt"
    from constant import Config
    config = Config()
for i,v in config.__dict__.items():
    logger.info(f"{i}: {v}")


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

if args.model_type == "ours":
    logger.info(f"meta_cfd_threshold:{config.meta_cfd_threshold}")


if config.test_top_feat == True:
    logger.info(f"Test by using single feature with threshold for selection!!!")
    logger.info(f"feature:{config.test_single_feat_name}")
    logger.info(f"Top rate:{config.top_rate}")

if config.test_top_feat == False and config.test_single_feat == True:
    logger.info("Train our meta model on single feature!!!")
    logger.info(f"feature:{config.test_single_feat_name}")



dirs = [train_dir,dev_dir,meta_dev_dir,unlabel_dir,test_dir]
datas = [None] * len(dirs)

# Build tokenizer and model
if config.pretrained_model == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num,output_hidden_states=True)
elif config.pretrained_model == "distill":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=config.class_num,output_hidden_states=True)
elif config.pretrained_model == "distill2":
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = BertForSequenceClassification.from_pretrained("distilroberta-base",num_labels=config.class_num,output_hidden_states=True)


def readandwrite(manifest,write_dir):
    file = open(write_dir, "wb")
    data = {}
    with open(manifest,'r',encoding='utf8') as f:
        for line in f.readlines():
            info = json.loads(line.strip())
            tmp = tokenizer._tokenize(info['src'])
            info['len'] = len(tmp)
            data[int(info["index"])] = info
    pickle.dump(data, file)
    file.close()
for i,tdir in enumerate(dirs):
    wt_dir = tdir.replace(".txt",".pkl")
    # readandwrite(tdir,wt_dir)
    a_file = open(wt_dir, "rb")
    datas[i] = pickle.load(a_file)
    a_file.close()

label_mapping_ag = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
label_mapping_sms = {"ham": 0, "spam": 1}
label_mapping_youtube = {"ham": 0, "spam": 1}
label_mapping_trec = {"DESC": 0, "ENTY": 1, "HUM": 2, "ABBR": 3, "LOC": 4, "NUM": 5}
label_mapping_imdb = {"pos": 1, "neg": 0}
label_mapping_dict = {"imdb":label_mapping_imdb, "ag_news":label_mapping_ag, "sms":label_mapping_sms, "trec":label_mapping_trec, "imdb": label_mapping_imdb, 
"youtube":label_mapping_youtube, "youtube_2":label_mapping_youtube, "youtube_5":label_mapping_youtube, "youtube_20":label_mapping_youtube}
label_mapping = label_mapping_dict[args.data_set]


#################
# move: dev, train -> unlabel, 20:20:20
def data2unlabel(c_target, data, unlabel):
    data_index = list(sorted(data.keys()))
    data_index_max = data_index[-1]
    unlabel_index = list(sorted(unlabel.keys()))
    unlabel_index_max = unlabel_index[-1]
    data_ref = deepcopy(data)
    data_new = {}
    cnt = 0
    for i in range(len(data_ref)):
        d = data_ref[i]
        if c_target[d['original_target']] > 0:
            c_target[d['original_target']] -= 1 
            d["index"] = cnt
            data_new[cnt] = d
            cnt += 1
        else: 
            unlabel_index_max += 1
            d["index"] = unlabel_index_max
            unlabel[unlabel_index_max] = d
    return c_target, data_new, unlabel

train_set = DatasetXL(datas[0],label_mapping,tokenizer)
dev_set = DatasetXL(datas[1],label_mapping,tokenizer)
meta_dev_set = DatasetXL(datas[2],label_mapping,tokenizer)
unlabel_set = DatasetXL(datas[3],label_mapping,tokenizer)
test_set = DatasetXL(datas[4],label_mapping,tokenizer)
dev = deepcopy(datas[1])
meta = deepcopy(datas[2])

if args.test_few == True:
    logger.info(f"We will leave {args.num_few} samples for training set, meta set and dev set !!!")
    train = datas[0]
    dev = datas[1]
    meta = datas[2]
    unlabel = datas[3]
    test = datas[4]
    ratio_train = 2*args.num_few / len(train)
    ratio_dev = args.num_few / len(dev)
    ratio_meta = args.num_few / len(meta)
    c_ori_train = Counter([d['original_target'] for i,d in train.items()])
    c_ori_dev = Counter([d['original_target'] for i,d in dev.items()])
    c_ori_meta = Counter([d['original_target'] for i,d in meta.items()])
    c_ori_unlabel = Counter([d['original_target'] for i,d in unlabel.items()])
    c_ori_test = Counter([d['original_target'] for i,d in test.items()])

    logger.info("The composition of original train set:")
    logger.info(c_ori_train)
    logger.info("The composition of original dev set:")
    logger.info(c_ori_dev)
    logger.info("The composition of original meta set:")
    logger.info(c_ori_meta)
    logger.info("The composition of original unlabel set:")
    logger.info(c_ori_unlabel)
    logger.info("The composition of original test set:")
    logger.info(c_ori_test)
    c_target_train = {i:math.ceil(ratio_train * v) for i,v in c_ori_train.items()}
    c_target_dev = {i:math.ceil(ratio_dev * v) for i,v in c_ori_dev.items()}
    if args.data_set == "trec":
        c_target_meta = {i:round(ratio_meta * v) for i,v in c_ori_meta.items()}
    else:
        c_target_meta = {i:math.ceil(ratio_meta * v) for i,v in c_ori_meta.items()} 
    logger.info("The composition of current train set:")
    logger.info(c_target_train)
    logger.info("The composition of current dev set:")
    logger.info(c_target_dev)
    logger.info("The composition of current meta set:")
    logger.info(c_target_meta)
    new_train = {}
    train_index = list(sorted(train.keys()))
    c_target_train, train, unlabel = data2unlabel(c_target_train, train, unlabel)
    c_target_dev, dev, unlabel = data2unlabel(c_target_dev, dev, unlabel)
    c_target_meta, meta, unlabel = data2unlabel(c_target_meta, meta, unlabel)

    
    train_set = DatasetXL(train,label_mapping)
    dev_set = DatasetXL(dev,label_mapping)
    meta_dev_set = DatasetXL(meta,label_mapping)
    unlabel_set = DatasetXL(unlabel,label_mapping)
    test_set = DatasetXL(datas[4],label_mapping)


# save few-shot data for use
data_dir = f"/home/self-training/ours/dataset/{args.data_set}_few/"

pickle.dump(train,open(data_dir+"train_data.pkl","wb"))
pickle.dump(datas[4],open(data_dir+"test_data.pkl","wb"))
pickle.dump(unlabel,open(data_dir+"unlabeled_data.pkl","wb"))
pickle.dump(dev,open(data_dir+"dev_data.pkl","wb"))
pickle.dump(meta,open(data_dir+"meta_dev_data.pkl","wb"))


meta = deepcopy(datas[2])
dev = deepcopy(datas[1])
max_idx_meta = len(meta)
max_idx_dev = len(dev)
for i in range(max_idx_meta):
    sample = meta.pop(i)
    sample['index'] = max_idx_dev
    dev[max_idx_dev] = sample
    max_idx_dev += 1
total_dev_set = DatasetXL(dev,label_mapping)

logger.info("#"*100)
logger.info("Dataset Statistics")
logger.info("Train_size:{};Test_size:{};Unlabeled_size:{};Dev_size:{};Meta_dev_size:{};Total_dev_size:{}".format(train_set.__len__(), \
test_set.__len__(), unlabel_set.__len__(), dev_set.__len__(),meta_dev_set.__len__(),total_dev_set.__len__()))
logger.info("#"*100)




# Criterion & optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) #or AdamW

# Init Trainer
# args.save_path = "/home/self-training/ours/output_youtube/0_baseline_20211027-094959/sup"
trainer = Trainer(config, model,logger, loss_function, optimizer, 
                  args.save_path, train_set,dev_set,meta_dev_set,total_dev_set,unlabel_set,test_set,label_mapping,
                  args.model_type, args.do_augment)


end_init = time.time()

logger.info('#'*100)
logger.info("Begin Our Self-training!!!")
logger.info('#'*100)
# !!! self-training
cfd = 0
if args.model_type == 'baseline':
    logger.info('#'*100)
    if config.baseline == 'naive':
        logger.info("Baseline: Naive Self-training!!!")
        logger.info('#'*100)
        trainer.naive_self_train(load=False)
    elif config.baseline == 'thres':
        logger.info("Baseline: Basic Self-training with threshold!!!")
        logger.info('#'*100)
        trainer.thres_self_train(load=False)
    else:
        logger.info("ERROR!YOU MUST CHOOSE BETWEEN {naive, thres}!")
else:
    trainer.self_train(load=False,early_stop=True, check_signal=False)

end_self = time.time()
if args.model_type == 'baseline':
    logger.info("Finish Naive Self-training!")
else:
    logger.info("Finish Our Self-training!")
logger.info("Self_Training_time:{:.2f}s".format(end_self-end_init))

# eval semi-supervised trained model 
epoch_loss, epoch_accu, preds_labels_ids = trainer.evaluator.evaluate(trainer.global_best_model, trainer.test_loader, is_test=True, return_details=True)
end_eval = time.time()
logger.info("Finish Evaluation on self-training!")
logger.info("Self_Training_Evaluation_time:{:.2f}s".format(end_eval-end_self))

with open(args.save_path + "/preds_labels_test_{}.pkl".format(args.model_type),"wb") as f:
    pickle.dump(preds_labels_ids, f)

label_mapping_rev = {key:value for value,key in label_mapping.items()}
with open(args.save_path + "/preds_labels_test_{}_{}.txt".format(args.model_type,args.num),"w") as f:
    i = 0
    for pred, target, id in preds_labels_ids:
        line = {"index":int(id), "text":test_set[int(id)], "pred":label_mapping_rev[int(pred)], "target":label_mapping_rev[int(target)]}
        # pickle.dump(line,f)
        f.write(str(line)+'\n')
