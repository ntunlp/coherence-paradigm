import torch
import load_data
import transformers
import model
import pickle
import torch.nn as F
import os
import sys
import time
import torch.nn as nn
from args import parser
from transformers import XLNetModel, XLNetTokenizer
from tqdm import tqdm

class EvalModel():
        def __init__(self, args):
                self.batch_size = args.batch_size
                self.model_size = args.model_size
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.test_file = args.test_file
                self.negs = args.num_negs
                self.margin = args.margin
                self.datatype = args.data_type
                self.max_len = args.max_len

                self.xlnet_model = model.MIM(args, self.device)

                pretrained = torch.load(args.pretrained_model)
                self.xlnet_model.load_state_dict(pretrained, strict=False)
                self.xlnet_model = self.xlnet_model.to(self.device)

        def test_model(self):
                self.xlnet_model.eval()
                test_data = load_data.LoadConnData(self.test_file, self.batch_size, self.model_size, self.device, self.datatype, self.negs, self.max_len)
                test_loader = test_data.data_loader()

                correct = 0.0
                total = 0.0
                start = time.asctime(time.localtime(time.time()))
                scores = []

                print("TESTING START: {}\n".format(start))
                with torch.no_grad():
                        for data in tqdm(test_loader):
                                try:
                                        pos_input, neg_inputs = data
                                except Error as e:
                                        print(e)
                                        continue

                                pos_score, neg_scores = self.xlnet_model.eval_forward(pos_input, neg_inputs)
                                max_neg_score = max(neg_scores)#, -1).values()
                                #print(pos_score, neg_scores)
                                temp = {}
                                temp['pos'] = pos_score
                                temp['neg'] = max_neg_score
                                scores.append(temp)
                                #max_neg_score = max(neg_scores)#, -1).values()
                                #max_neg_score = torch.max(neg_scores, -1).values()

                                if pos_score > max_neg_score:
                                        correct += 1.0
                                
                                total += 1.0

                pickle.dump(scores, open('temp-eval-dump', 'wb'))
                acc = correct/total
                end = time.asctime(time.localtime(time.time()))
                print("TESTING END: {} ACC: {}\n".format(end, acc))
                
                return


args = parser.parse_args()
Eval = EvalModel(args)
Eval.test_model()
