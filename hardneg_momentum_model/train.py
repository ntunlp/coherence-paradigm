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
import random
from args import parser
from transformers import AdamW
from torch.optim.swa_utils import SWALR
from transformers import XLNetModel, XLNetTokenizer
from tqdm import tqdm

class TrainModel():

    
    def save_model(self, step, accuracy):
        if not os.path.isdir('saved_models'):
            os.mkdir('saved_models')
        model_path = os.path.join("saved_models", "{}_seed-{}_bs-{}_lr-{}_step-{}_type-{}_acc-{}.mom".format(self.desc, self.seed, self.batch_size, self.learning_rate, step, self.model_size, accuracy))
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
        self.rank_negs = args.num_rank_negs
        self.train_steps = args.train_steps
        self.margin = args.margin
        self.desc = args.model_description
        self.seed = args.seed
        self.datatype = args.data_type
        self.max_len = args.max_len
        self.bestacc = 0.0

        self.xlnet_model = model.MIM(args, self.device)
        
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

    def get_ranked_negs(self, neg_scores):
        ranked_idx = sorted(range(len(neg_scores)), key=neg_scores.__getitem__, reverse=True)
        hard_negs = ranked_idx[:self.negs]
        return hard_negs


    def get_next_train_data(self, processed_exploration_data):
        self.xlnet_model.eval()

        next_train_data = []
        with torch.no_grad():
            for i, each_data in enumerate(processed_exploration_data):
                try:
                    pos_input, slice_input, neg_input = each_data
                except Error as e:
                    print(e)
                    continue

                pos_score, neg_scores = self.xlnet_model.eval_forward(pos_input, neg_input)
                pos_score = pos_score.to(torch.device("cpu"))
                neg_scores = neg_scores.to(torch.device("cpu"))

            
                next_neg_idx = self.get_ranked_negs(neg_scores)

                if len(next_neg_idx) < self.negs:
                    continue

                neg_data_list = torch.stack([neg_input[0][x] for x in next_neg_idx]).unsqueeze(0)
                next_train_data.append([pos_input, slice_input, neg_data_list])
        
        return next_train_data



    def hard_negs_controller(self):
        train_data = load_data.ConnDataset(self.train_file, self.model_size, self.device, self.datatype, self.negs, self.max_len)
        init_train_data = train_data.data[:self.train_steps]
        total_iterations = len(train_data.data)//self.train_steps 
       
        for iteration_index in range(total_iterations):
            full_time = time.asctime(time.localtime(time.time()))
            
            print("ITERATION: {} TIME: {} LOSS: {}".format(iteration_index, full_time, self.total_loss))
            self.total_loss = 0.0

            if iteration_index == 0:
                processed_train_data_list = train_data.prepareTrainData(init_train_data, self.negs)
                self.train_xlnet_model(processed_train_data_list, iteration_index)
                next_train_data = []
            else:
                start_index = iteration_index*self.train_steps
                end_index = start_index + self.train_steps
                
                processed_explore_data_list = train_data.prepareTrainData(train_data.data[start_index:end_index], self.rank_negs)
                next_train_data = self.get_next_train_data(processed_explore_data_list)
                self.train_xlnet_model(next_train_data, iteration_index)

                if (self.train_steps*(iteration_index+1))%self.eval_interval==0:
                    self.scheduler.step()
                    self.eval_model(self.dev_file, self.train_steps*(iteration_index+1), start)
        
        self.eval_model(self.dev_file, self.train_steps*(iteration_index+1), start)



    def train_xlnet_model(self, train_loader, iteration_index):    
        start = time.time()
        self.xlnet_model.train()

        for step, data in enumerate(train_loader):
            
            self.optimizer.zero_grad()

            try:
                pos_input, slice_input, neg_input = data
            except Error as e:
                print(e)
                continue


            combined_loss = self.xlnet_model(pos_input, slice_input, neg_input)
            combined_loss.backward()
            
            self.xlnet_model.updateMomentumEncoder()
            self.optimizer.step()
        
            self.total_loss += combined_loss.item()

    def eval_model(self, data_file, step, start):
        
        print(self.desc, self.seed, "EVAL START")
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

                pos_score, neg_scores = self.xlnet_model.eval_forward(pos_input, neg_inputs)
                try:
                    max_neg_score = torch.max(neg_scores, -1).values
                except:
                    max_neg_score = max(neg_scores)
                
                if pos_score > max_neg_score:
                    correct += 1.0
                total += 1.0

        self.xlnet_model.train()
        end = time.time()
        full_time = time.asctime(time.localtime(end))
        acc = correct/total
        if data_file == self.dev_file:
            print("DEV EVAL Time: {} Elapsed: {} Steps: {} Acc: {}".format(full_time, end-start, step,  acc))
            if step > 0:
                self.bestacc = acc
                self.save_model(step, acc)
        elif data_file == self.test_file:
            print("Please evaluate the test file separately with the best saved checkpoint.")
            print("TEST EVAL Time: {} Steps: {} Acc: {}".format(full_time, end-start, step,  acc))

        return
            
                



args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

start = time.time()

Trainer = TrainModel(args)
Trainer.hard_negs_controller()


