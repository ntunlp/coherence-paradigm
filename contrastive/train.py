import torch
import load_data
import transformers
import model
import pickle
import torch.nn as F 
import time
import os
import sys
import datetime
import torch.nn as nn
from args import parser
from transformers import AdamW
from torch.optim.swa_utils import SWALR
from transformers import XLNetModel, XLNetTokenizer
from tqdm import tqdm


class TrainModel():

    def save_model(self, step, accuracy):
        if not os.path.isdir('saved_models'):
            os.mkdir('saved_models')
        model_path = os.path.join("saved_models", "{}_seed-{}_bs-{}_lr-{}_step-{}_acc-{}_type-{}.cont".format(self.desc, self.seed, self.batch_size, self.learning_rate, step, accuracy, self.model_size))

        torch.save(self.xlnet_model.state_dict(), model_path)
            


    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model_size = args.model_size
        self.learning_rate = args.lr_start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        if args.test_file:
            self.test_file = args.test_file
        else:
            self.test_file = args.dev_file
        self.negs = args.num_negs
        self.margin = args.margin
        self.desc = args.model_description
        self.seed = args.seed
        self.datatype = args.data_type
        self.anneal_to = args.lr_end
        self.anneal_epochs = args.lr_anneal_epochs
        self.bestacc = 0.0
        self.max_len = args.max_len
        

        self.xlnet_model = model.MIM(self.model_size, self.negs, self.batch_size, self.margin, self.device)
        try:
            if args.pretrained_model:
                pretrained = torch.load(args.pretrained_model)
                self.xlnet_model.load_state_dict(pretrained, strict=False)
                print("Loaded pretrained model")
        except:
            pass
       
        self.xlnet_model = self.xlnet_model.to(self.device)
                
        self.optimizer=AdamW(self.xlnet_model.parameters(), lr=self.learning_rate)
        self.scheduler = SWALR(self.optimizer, anneal_strategy="linear", anneal_epochs=args.lr_anneal_epochs, swa_lr=args.lr_end)
        self.total_loss = 0.0

        self.eval_interval = args.eval_interval


    def train_xlnet_model(self):    
        train_data = load_data.LoadConnData(self.train_file, self.batch_size, self.model_size, self.device, self.datatype, self.negs, self.max_len)
        train_loader = train_data.data_loader()
        start = time.time()
        self.xlnet_model.train()

        if 1:
            for step, data in enumerate(train_loader):
                
                self.optimizer.zero_grad()

                try:
                    pos_input, neg_input = data
                except Error as e:
                    print(e)
                    continue


                pos_score, neg_scores = self.xlnet_model(pos_input, neg_input)
                
                loss = self.xlnet_model.contrastiveLoss(pos_score, neg_scores)
            
                loss.backward()
                self.optimizer.step()

                self.total_loss += loss.item()

                if step%self.eval_interval == 0 and step > 0:
                    self.eval_model(self.dev_file, step, start)
                    self.scheduler.step()
                    
            
                if step%1000 == 0:
                    end = time.time()
                    full_time = time.asctime(time.localtime(end))
                    print("LOG Time: {} Elapsed: {} Steps: {} Loss: {}".format(full_time, end-start, step, loss.item()))

    
            self.eval_model(self.test_file, step, start)

    def eval_model(self, data_file, step, start):
        batch_size = self.batch_size
        self.xlnet_model.eval()
        test_data = load_data.LoadConnData(data_file, self.batch_size, self.model_size, self.device, self.datatype, self.negs, self.max_len)
        test_loader = test_data.data_loader()

        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for data in test_loader:
                try:
                    pos_input, neg_inputs = data
                except Error as e:
                    print(e)
                    continue

                pos_score, neg_scores = self.xlnet_model(pos_input, neg_inputs)
                max_neg_score = torch.max(neg_scores, -1).values
                
                if pos_score > max_neg_score:
                    correct += 1.0
                total += 1.0

        self.xlnet_model.train()
        end = time.time()
        full_time = time.asctime(time.localtime(end))
        acc = correct/total
        if data_file == self.dev_file:
            print(self.desc, self.seed)
            print("DEV EVAL Time: {} Elapsed: {} Steps: {} Acc: {}".format(full_time, end-start, step,  acc))
            if step > 0:
                self.bestacc = acc
                self.save_model(step, acc)
        elif data_file == self.test_file:
            print("Please evaluate the test set separately with the best saved checkpoint.")
            print("TEST EVAL Time: {} Elapsed: {} Acc: {}".format(full_time, end-start, step,  acc))


                
        return
            
                


args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
start = time.time()

Trainer = TrainModel(args)
Trainer.train_xlnet_model()
