import numpy as np
import copy
from sacreBLEU.sacreBLEU import corpus_bleu

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import spacy
import pdb
import os
from underthesea import word_tokenize
import jieba
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle as pkl
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# PREPROCESSING UTILS
# --------------------------------------------------------------------------------------------------------------
# :Lang: Language object that holds language dicts.
# :read_dataset: Read a translation dataset as a pandas dataframe. Switched to this to detect the misalignment.
# :data_tok: Tokenize sentences.
# :token2index_dataset: Creates token2index vocab for the passed dataset.
# :Vietnamese: New language object unique to Vietnamese.
# :Chinese: New langage object unique to Chinese.
# :translation_collate: collate function to use in the training dataloader.
# :translation_collate_val: collate function to use in the validation/test dataloader.
# ---------------------------------------------------------------------------------------------------------------

# Taken from lab notebook
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2:"<UNK>",3:"<PAD>"}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def read_dataset(path):
    f = open(path)
    list_l = []
    for line in f:
        list_l.append(line.strip())
    data = pd.DataFrame()
    data["data"] = list_l
    return data

# Taken from lab notebook
def unicodeToAscii(s):
    """About "NFC" and "NFD": 
    
    For each character, there are two normal forms: normal form C 
    and normal form D. Normal form D (NFD) is also known as canonical 
    decomposition, and translates each character into its decomposed form. 
    Normal form C (NFC) first applies a canonical decomposition, then composes 
    pre-combined characters again.
    
    About unicodedata.category: 
    
    Returns the general category assigned to the Unicode character 
    unichr as string."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def data_tok(data, lang="vi"):
    
    data["en_tokenized"] = data["en_data"].apply(lambda x: x.lower().split( ))
    
    if lang == "vi":
        data["vi_tokenized"] = data["vi_data"].apply(lambda x: x.lower().split( ))
    else:
        data["zh_tokenized"] = data["zh_data"].apply(lambda x: x.lower().split( ))
        
    return data


def token2index_dataset(data, source_language="zh"):
    
    if source_language == "zh" and "zh_data" not in [*data.columns]:
        raise ValueError, "Source language should be compatible with the data you pass!"
    elif source_language == "vi" and "vi_data" not in [*data.columns]:
        raise ValueError, "Source language should be compatible with the data you pass!"
    else:   
        if source_language == "zh":
            # chinese -> english
            for language in ["en","zh"]:
                indices_data = []
                if language == "en":
                    lang_obj = zhen_en_
                else:
                    lang_obj = zhen_zh_

                for tokens in data[language + "_tokenized"]:

                    index_list = [lang_obj.word2index[token] if \
                                  token in lang_obj.word2index else UNK_IDX \
                                  for token in tokens]
                    index_list.append(EOS_token)
                    indices_data.append(index_list)

                data[language + "_indices"] = indices_data
        else:
            # vietnamese -> english
            for language in ["en","vi"]:
                indices_data = []
                if language == "en":
                    lang_obj = vien_en_
                else:
                    lang_obj = vien_vi_

                for tokens in data[language + "_tokenized"]:

                    index_list = [lang_obj.word2index[token] if \
                                  token in lang_obj.word2index else UNK_IDX \
                                  for token in tokens]
                    index_list.append(EOS_token)
                    indices_data.append(index_list)

                data[language + "_indices"] = indices_data

    return data


# Datasets
from torch.utils.data import Dataset

# vietnamese -> english
class Vietnamese(Dataset):
    
    def __init__(self, data, val = False):
        self.data = data
        self.val = val
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        english = self.data.iloc[idx,:]["en_indices"]
        vietnamese = self.data.iloc[idx,:]["vi_indices"]
        en_lengths = self.data.iloc[idx,:]["en_lengths"]
        vi_lengths = self.data.iloc[idx,:]["vi_lengths"]
        
        if self.val:
            en_data = self.data.iloc[idx,:]["en_data"].lower()
            return [vietnamese, english, vi_lengths, en_lengths, en_data]
        else:
            return [vietnamese, english, vi_lengths, en_lengths]
    
# chinese -> english
class Chinese(Dataset):
    def __init__(self, data, val = False):
        self.data = data
        self.val = val
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        english = self.data.iloc[idx,:]["en_indices"]
        chinese = self.data.iloc[idx,:]["zh_indices"]
        en_lengths = self.data.iloc[idx,:]["en_lengths"]
        zh_lengths = self.data.iloc[idx,:]["zh_lengths"]
        
        if self.val:
            en_data = self.data.iloc[idx,:]["en_data"].lower()
            return [chinese, english, zh_lengths, en_lengths, en_data]
        else:
            return [chinese, english, zh_lengths, en_lengths]


MAX_LEN_TARGET = 50 # EN
MAX_LEN_SOURCE = 50 # CHINESE/VIETNAMESE

def translation_collate(batch):
    
    if MAX_LEN_TARGET <= 10:
        raise ValueError("MAX_LEN_TARGET too small")
    elif MAX_LEN_SOURCE <= 10:
        raise ValueError("MAX_LEN_SOURCE too small")
        
    else:
    
        target_sentence = []
        source_sentence = []
        target_lengths = []
        source_lengths = []

        for datum in batch:
            target_lengths.append(datum[3])
            source_lengths.append(datum[2])

        max_target_length = max(target_lengths)
        max_source_len = max(source_lengths)

        if max_target_length < MAX_LEN_TARGET:
            MAX_LEN_TARGET = max_target_length

        if max_source_len < MAX_LEN_SOURCE:
            MAX_LEN_SOURCE = max_source_len

        # padding
        for datum in batch:
            if datum[2] > MAX_LEN_SOURCE:
                padded_vec_source = np.array(datum[0])[:MAX_LEN_SOURCE]
            else:
                padded_vec_source = np.pad(np.array(datum[0]),
                                    pad_width=((0,MAX_LEN_SOURCE - datum[2])),
                                    mode="constant", constant_values=PAD_IDX)
            if datum[3] > MAX_LEN_TARGET:
                padded_vec_target = np.array(datum[1])[:MAX_LEN_TARGET]
            else:
                padded_vec_target = np.pad(np.array(datum[1]),
                                    pad_width=((0,MAX_LEN_TARGET - datum[3])),
                                    mode="constant", constant_values=PAD_IDX)
                
            target_sentence.append(padded_vec_target)
            source_sentence.append(padded_vec_source)

        source_sentence = np.array(source_sentence)
        target_sentence = np.array(target_sentence)
        source_lengths = np.array(source_lengths)
        target_lengths = np.array(target_lengths)

        source_lengths[source_lengths>MAX_LEN_SOURCE] = MAX_LEN_SOURCE
        target_lengths[target_lengths>MAX_LEN_TARGET] = MAX_LEN_TARGET

    return [torch.from_numpy(source_sentence), torch.from_numpy(target_sentence),
            torch.from_numpy(source_lengths), torch.from_numpy(target_lengths)]


def translation_collate_val(batch):
	# batch_size is always 1 for val and test, as we are computing 
	# corpus-level BLEU. 
    return [torch.from_numpy(np.array(batch[0][0])).unsqueeze(0), 
            torch.from_numpy(np.array(batch[0][1])).unsqueeze(0),
            torch.from_numpy(np.array(batch[0][2])).unsqueeze(0), 
            torch.from_numpy(np.array(batch[0][3])).unsqueeze(0),batch[0][4]]




