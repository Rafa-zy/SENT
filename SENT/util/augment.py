import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm_notebook as tqdm
import os
import re
import pickle
import random
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Load translation model
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')

# Load an En-De Transformer model trained on WMT'19 data:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe').to('cuda')

stop_words = set(stopwords.words('english'))


def back_translate_ru(text):
    hyp_list = [0.7, 0.8, 0.9]
    hyp = random.choice(hyp_list)
    ru = en2ru.translate(text, sampling=True, temperature=hyp)
    return ru2en.translate(ru, sampling=True, temperature=hyp)


def back_translate_de(text):
    hyp_list = [ 0.7, 0.8, 0.9]
    hyp = random.choice(hyp_list)
    de = en2de.translate(text, sampling=True, temperature=hyp)
    return de2en.translate(de, sampling=True, temperature=hyp)


def back_translate(texts, labels):
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        augmented_texts.append(back_translate_ru(text))
        augmented_labels.append(label)
        augmented_texts.append(back_translate_de(text))
        augmented_labels.append(label)
    return augmented_texts, augmented_labels 


def back_translate_random_sampling():
    pass


def aug_backtranslate(text_list, targets):
    print('aug_backtranslate')
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(text_list, targets):
        print('origin text', text, label)
        sents = paragraph_to_sentences(text)
        label = label.item()
        
        back_tr_sents = []
        for sent in sents:
            back_tr_sent = back_translate_ru(sent)
            back_tr_sents.append(back_tr_sent)
        
        back_tr_sents = sentences_to_pargraph(back_tr_sents)
        print('back_tr_sents', back_tr_sents)
        print()
        augmented_texts.append(back_tr_sents)
        augmented_labels.append(label)
    assert len(augmented_texts) == len(augmented_labels)
    
    return augmented_texts, augmented_labels


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if '_' in l.name() or '-' in l.name():
                continue
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)


def synonym_replacement(words, num):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >=1 :
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= num:
            break
            
    sentence = ' '.join(new_words)
    return sentence


def word_replacement(texts, labels):
    replaced_texts = []
    replaced_labels = []
    replace_ratio = 0.3
    
    for sent, label in zip(texts, labels):
        words = sent.split(' ')
        num_words = len(words)
        n_sr = max(1, int(replace_ratio*num_words))
        replaced_sent = synonym_replacement(words, n_sr)
        replaced_texts.append(replaced_sent)
        replaced_labels.append(label)        
    return replaced_texts, replaced_labels
    
    
import nltk


def sentences_to_pargraph(sents_list):
    return ' '.join(sents_list)

def paragraph_to_sentences(text):
    sent_tokenizer = nltk.tokenize.sent_tokenize
    text = text.strip()
    sent_list  = sent_tokenizer(text)
    has_long = False
    for split_punc in [".", ";", ",", " ", ""]:
        if split_punc == " " or not split_punc:             
            offset = 100
        else:
            offset = 5
        has_long = False
        new_sent_list = []
        for sent in sent_list :
            if len(sent) < 300:
                new_sent_list += [sent]
            else:
                has_long = True
                sent_split = split_sent_by_punc(sent, split_punc, offset)
                new_sent_list += sent_split
        sent_list = new_sent_list
        if not has_long :
            break
    return sent_list


def split_sent_by_punc(sent, punc, offset):
    """Further split sentences when nltk's sent_tokenizer fail."""
    sent_list = []
    start = 0
    while start < len(sent):
        if punc:
            pos = sent.find(punc, start + offset)
        else:
            pos = start + offset
        if pos != -1:
            sent_list += [sent[start: pos + 1]]
            start = pos + 1
        else:
            sent_list += [sent[start:]]
            break
    return sent_list
