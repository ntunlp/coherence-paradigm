import torch
#import train
import pickle
import sys
import pandas as pd
import string
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

random.seed(100)

class ConnDataset(Dataset):
    def __init__(self, fname, model, device, datatype, negs, max_len):
        self.fname = fname
        self.device = device
        self.data = pickle.load(open(fname, 'rb'))
        random.shuffle(self.data)
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


    def getSlice(self, doc):
        try:
            end = random.choice(range(4,len(doc)))
            return doc[:end]
        except:
            return doc


    def prepareTrainData(self, data_list, num_negs):
        train_list = []
        for each_item in data_list:
            train_list.append(list(self.prepareEachItem(each_item, num_negs)))
        return train_list


    def prepareEachItem(self, train_data_item, num_negs):
        pos_doc = train_data_item['pos']
        neg_docs = train_data_item['negs'][:num_negs]

        pos_span = pos_doc
        pos_span = ' '.join(pos_span)
        pos_tokens = self.tokenizer.tokenize(pos_span)
        pos_ids = self.tokenizer.convert_tokens_to_ids(pos_tokens)
        pos_ids = self.pad_ids(pos_ids)

        pos_slice = ' '.join(self.getSlice(pos_doc))
        slice_tokens = self.tokenizer.tokenize(pos_slice)
        slice_ids = self.tokenizer.convert_tokens_to_ids(slice_tokens)
        slice_ids = self.pad_ids(slice_ids)

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
        slice_input = self.tokenizer.build_inputs_with_special_tokens(slice_ids)

        pos_tensor = torch.tensor(pos_input).unsqueeze(0).to(self.device)
        slice_tensor = torch.tensor(slice_input).unsqueeze(0).to(self.device)
        neg_tensor_stack = torch.stack(neg_span_list).unsqueeze(0).to(self.device)
        
        return pos_tensor, slice_tensor, neg_tensor_stack
                

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

