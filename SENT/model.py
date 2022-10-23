
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from transformers import BertTokenizer
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear, MetaEmbedding, MetaLayerNorm
# from metalinear_ws import MetaWeightSharedLinear

class BERT_META(MetaModule):
    def __init__(self, num_labels):
        super(BERT_ATTN, self).__init__()

        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        # self.init_weights()
        self.attention_mask = None
    
    def attention(self, pooler_output, last_hidden_state):
        pooler_output = pooler_output.unsqueeze(1)
        last_hidden_state = last_hidden_state.transpose(1,2)

        attn = torch.bmm(pooler_output, last_hidden_state)
        attn = attn.squeeze(1)
        
        if self.attention_mask is not None:
            mask = (1-self.attention_mask).type(torch.BoolTensor).to('cuda')
            attn.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)
        
        return attn
    
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        last_hidden_state, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        
        # set attention mask
        self.attention_mask = attention_mask
        self.attention_mask[:, 0] = 0 # mask cls token 
        sep_mask = (input_ids == self.tokenizer.sep_token_id) # mask sep token
        self.attention_mask.masked_fill_(sep_mask, 0)
        
        attn = self.attention(pooler_output, last_hidden_state)
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits, attn)

class BERT_ATTN(torch.nn.Module):
    
    def __init__(self, num_labels):
        super(BERT_ATTN, self).__init__()

        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        # self.init_weights()
        self.attention_mask = None
    
    def attention(self, pooler_output, last_hidden_state):
        pooler_output = pooler_output.unsqueeze(1)
        last_hidden_state = last_hidden_state.transpose(1,2)

        attn = torch.bmm(pooler_output, last_hidden_state)
        attn = attn.squeeze(1)
        
        if self.attention_mask is not None:
            mask = (1-self.attention_mask).type(torch.BoolTensor).to('cuda')
            attn.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)
        
        return attn
    
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        last_hidden_state, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        
        # set attention mask
        self.attention_mask = attention_mask
        self.attention_mask[:, 0] = 0 # mask cls token 
        sep_mask = (input_ids == self.tokenizer.sep_token_id) # mask sep token
        self.attention_mask.masked_fill_(sep_mask, 0)
        
        attn = self.attention(pooler_output, last_hidden_state)
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits, attn)
        
class MLP_head(nn.Module):
    def __init__(self,hid_size,input_size,output_size,act="elu"):
        super(MLP_head, self).__init__()
        self.hidden_size = hid_size
        self.input_size = input_size
        self.output_size = output_size
        self.act_type = act
        if self.hidden_size > 0:
            self.linear1 = nn.Linear(self.input_size,self.hidden_size)
            if self.act_type == "elu":
                self.act=nn.ELU()
            else:
                self.act = nn.ReLU()
            self.linear2 = nn.Linear(self.hidden_size,self.output_size)
        else:
            self.linear2 = nn.Linear(self.input_size,self.output_size)


    def forward(self,x,padding_mask=None,softmax=False):
        if self.hidden_size > 0:
            x = self.linear1(x)
            x = self.act(x)
        out = self.linear2(x)
        if softmax:
            out = F.softmax(out,dim=-1)
        if padding_mask is not None:
            out = out * (padding_mask)
        return out

class StochasticDepth(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, p: float = 0.5):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError(
                "Stochastic Depth p has to be between 0 and 1 but got {}".format(p)
            )
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_():
            return inputs
        return self.p * self.module(inputs)

class Linear_Main(nn.Module):
    def __init__(self,emb_size,hidden_size,num_layers,num_labels,dropout=0.5):
        super(Linear_Main, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.dropout = dropout
        modules = [nn.Dropout(self.dropout)]
        for i in range(self.num_layers):
            if i == 0:
                modules.append(nn.Linear(self.emb_size,self.hidden_size))
            else:
                modules.append(nn.Linear(self.hidden_size,self.hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))
        if self.num_layers > 0:
            modules.append(nn.Linear(self.hidden_size,self.num_labels))
        else:
            modules.append(nn.Linear(self.emb_size,self.num_labels))
        self.linears = nn.ModuleList(modules)

    def forward(self,x):
        out = x
        for mod in self.linears:
            out = mod(out)
        return out





class Linear_Noisy(nn.Module):
    def __init__(self,emb_size,hidden_size,num_layers,num_labels,dropout=0.5):
        super(Linear_Noisy, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.dropout = dropout
        modules = [nn.Dropout(self.dropout)]
        for i in range(self.num_layers):
            if i == 0:
                mod = nn.Linear(self.emb_size,self.hidden_size)
                mod_noisy = StochasticDepth(mod,p=0.5)
                modules.append(mod_noisy)
            else:
                mod = nn.Linear(self.hidden_size,self.hidden_size)
                mod_noisy = StochasticDepth(mod,p=0.5)
                modules.append(mod_noisy)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))
        if self.num_layers > 0:
            modules.append(nn.Linear(self.hidden_size,self.num_labels))
        else:
            modules.append(nn.Linear(self.emb_size,self.num_labels))
        self.linears = nn.ModuleList(modules)
        # self.noisy_linears = StochasticDepth(self.linears,p=0.5)

    def forward(self,x):
        out = x
        for mod in self.linears:
            out = mod(out)
        return out