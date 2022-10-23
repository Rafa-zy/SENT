import torch
class Config():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = 2021
        self.training_from_scratch = True # suitable for both baseline and ours
        self.baseline = "naive"
        self.pretrained_model = "bert"
        self.learning_rate = 2e-5 # 2e-5
        self.epochs = 10 # 10
        self.inner_epochs = 1 # 10
        self.meta_epochs = 200 # 200
        self.hist_len = 100 # 100
        self.multiple_len = 1
        self.main_target='acc'
        self.meta_target='precision'
        self.beta = 0.25
        self.confidence_thres = 0.0
        self.add_meta_ratio = 0
        self.block_entropy = False
        self.dropout = 0.1

        self.meta_cfd_threshold = 0.8
        self.relative = False

        self.split_num_groups = 5
        self.train_max_token_len = 15000
        self.train_batch_size = 2 # 16
        self.valid_batch_size = 2 # 16
        self.valid_max_token_len = 15000
        self.test_batch_size = 16 # 128
        self.unlabeled_batch_size = 16 # 128
        self.class_num = 2
        self.sample_num = 125 # 500 <-> 1:12.5
        self.label_sample_ratio_imdb = 0.02
        self.dev_sample_ratio_imdb = 0.5
        self.meta_dev_sample_ratio_imdb = 0.5

        self.label_sample_ratio_sms = 0.02
        self.dev_sample_ratio_sms = 0.5
        self.meta_dev_sample_ratio_sms = 0.5

        self.label_sample_ratio_trec = 0.02
        self.dev_sample_ratio_trec = 0.5
        self.meta_dev_sample_ratio_trec = 0.5

        self.label_sample_ratio_youtube = 0.1
        self.dev_sample_ratio_youtube = 0.5
        self.meta_dev_sample_ratio_youtube = 0.5


        self.bidirectional = True

        #meta model
        self.hidden_size = 128
        self.act = 'elu'

        self.meta_lr = 1e-3
        self.meta_lr_min = 1e-6
        self.patience = 10
        self.meta_patience = 5

        self.loss_gamma = 0.9

        # feature_engineering
        self.use_loss = True
        self.use_ema_loss = True
        self.use_hist_entropy = True
        self.use_entropy = True
        self.use_simi = True
        self.soft_entropy = False
        self.soft_entropy_mul = False
        self.input_size = 0
        self.test_single_feat = False
        self.feature_pos_map = {"hist_emaloss":True,"cur_loss":True,"hist_entropy":True,"cur_entropy":True,"cur_simi":False}
        self.test_single_feat_name = "hist_entropy" #
        self.test_top_feat = False
        self.top_rate = 0.2
        self.positive = self.feature_pos_map[self.test_single_feat_name]

        # co-teaching
        self.co_teaching_plus = True
        self.co_epochs = 50
        self.forget_rate = 0.2
        self.forget_schedule = 500
        self.T_k = 5
        self.max_step = 10000 # 5000
        self.warmup_step = 1000
        self.eval_interval = 100
        self.log_interval = 50
        self.lr_min = 1e-6