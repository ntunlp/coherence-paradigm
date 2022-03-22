import torch

import pickle
import sys
import pandas as pd
import string
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler



class ConnDataset(Dataset):
    def __init__(self, fname, model, device, datatype, negs, max_len):
        self.fname = fname
        self.device = device
        self.data = pickle.load(open(fname, 'rb'))
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-{}-cased'.format(model))
        self.truncount = 0
        self.datatype = datatype
        self.negs = negs
        self.max_len = max_len

    def pad_ids(self,ids):
        if len(ids) < self.max_len:
            padding_size = self.max_len - len(ids)
            padding = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) for i in range(padding_size)]
            ids = ids + padding
        else:
            ids = ids[:self.max_len]
            self.truncount += 1

        return ids

    def prepareData(self, idx):
        pos_doc = self.data[idx]['pos']
        
        if self.datatype == 'single':
            neg_docs = [self.data[idx]['neg']]
        elif self.datatype == 'multiple':
            neg_docs = self.data[idx]['negs'][:self.negs]

        pos_span = pos_doc
        pos_span = ' '.join(pos_span)
        pos_tokens = self.tokenizer.tokenize(pos_span)
        pos_ids = self.tokenizer.convert_tokens_to_ids(pos_tokens)
        pos_ids = self.pad_ids(pos_ids)

        neg_span_list = []
        for neg_doc in neg_docs:    
            neg_span = neg_doc
            neg_span = ' '.join(neg_span)
            neg_tokens = self.tokenizer.tokenize(neg_span)
            neg_ids = self.tokenizer.convert_tokens_to_ids(neg_tokens)
            neg_ids = self.pad_ids(neg_ids)
            neg_input = self.tokenizer.build_inputs_with_special_tokens(neg_ids)
            
            neg_span_list.append(torch.tensor(neg_input))

        pos_input = self.tokenizer.build_inputs_with_special_tokens(pos_ids)
        
        return torch.tensor(pos_input).to(self.device), torch.stack(neg_span_list).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.prepareData(idx)


class LoadConnData():
    def __init__(self, fname, batch_size, model, device, datatype, negs, max_len):
        self.fname = fname
        self.batch_size = batch_size
        self.dataset = ConnDataset(fname, model, device, datatype, negs, max_len)

    def data_loader(self):
        dataSampler = SequentialSampler(self.dataset)
        loader = DataLoader(dataset=self.dataset, sampler=dataSampler, batch_size=self.batch_size)
        return loader
