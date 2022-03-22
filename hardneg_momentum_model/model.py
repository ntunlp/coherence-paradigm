import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel, XLNetLMHeadModel, XLNetTokenizer


class MIM(nn.Module):
        def __init__(self, args, device):

                super().__init__()
                self.batch_size = args.batch_size
                self.device = device
                self.m = args.momentum_coefficient #0.9999999 #momentum coefficient
                
                self.main_encoder = XLNetModel.from_pretrained('xlnet-{}-cased'.format(args.model_size))
                self.momentum_encoder = XLNetModel.from_pretrained('xlnet-{}-cased'.format(args.model_size))
                
                if args.model_size=='base':
                        hidden_size = 768
                elif args.model_size=='large':
                        hidden_size = 1024

                self.queue = []
                self.queue_size = args.queue_size
                self.con_loss_weight = args.contrastive_loss_weight
                self.negs = args.num_negs
                self.margin = args.margin
                self.cosim = nn.CosineSimilarity()
                self.crossEntropy = nn.CrossEntropyLoss()
                self.getTranspose = lambda x: torch.transpose(x, -2, -1)
                self.subMargin = lambda z: z - args.margin

                self.conlinear = nn.Linear(hidden_size, 1)
        
        def getMainScore(self, doc):
                rep = self.main_encoder(input_ids=doc).last_hidden_state[:,-1,:]
                score = self.conlinear(rep).view(-1)
                return score

        def getMomentumRep(self, doc):
                rep = self.momentum_encoder(input_ids=doc).last_hidden_state[:,-1,:]
                return rep.detach()

        def getSim(self, pos_rep, pos_slice):
            pos_sim = torch.matmul(pos_rep, self.getTranspose(pos_slice))
            neg_sims = torch.matmul(pos_rep, self.getTranspose(torch.stack(self.queue))).view(1,-1)
            all_sims = torch.cat((pos_sim, neg_sims), dim=1)
            return all_sims

        def getCosSim(self, pos_rep, pos_slice):
            pos_sim = self.cosim(pos_rep, pos_slice)
            neg_sims = [self.cosim(pos_rep, neg_x.view(1,-1)) for neg_x in self.queue]
            return pos_sim, neg_sims

        def updateMomentumEncoder(self):
            with torch.no_grad():
                for main, moco in zip(self.main_encoder.parameters(), self.momentum_encoder.parameters()):
                    moco.data = (moco.data * self.m) + (main.data * (1 - self.m))

                
        def forward(self, pos_doc, pos_slice, neg_docs):
                pos_rep = self.main_encoder(input_ids=pos_doc).last_hidden_state[:,-1,:]
                pos_score = self.conlinear(pos_rep).view(-1)
                
                pos_slice_rep = self.getMomentumRep(pos_slice) 
                        
                neg_scores = list(map(self.getMainScore, list(neg_docs)))
                neg_moco_rep = list(map(self.getMomentumRep, list(neg_docs)))
                
                if len(self.queue) >= self.queue_size: #global negative queue size
                    del self.queue[:self.negs]
                self.queue.extend(neg_moco_rep[0])

                all_sims = self.getSim(pos_rep, pos_slice_rep)
                pos_sim, neg_sims = self.getCosSim(pos_rep, pos_slice_rep)
                
                simContraLoss = self.simContrastiveLoss(pos_sim, neg_sims)
                contraLoss = self.contrastiveLoss(pos_score, neg_scores[0])

                full_loss = (self.con_loss_weight * contraLoss) + ((1-self.con_loss_weight) * simContraLoss)

                return full_loss

        def eval_forward(self, pos_doc, neg_docs):
            pos_score = self.getMainScore(pos_doc)
            neg_scores = torch.stack(list(map(self.getMainScore, list(neg_docs))))
            return pos_score.detach(), neg_scores[0].detach()

        def simContrastiveLoss(self, pos_sim, neg_sims):
            neg_sims_sub = torch.stack(list(map(self.subMargin, neg_sims))).view(-1)
            all_sims = torch.cat((neg_sims_sub, pos_sim), dim=-1)
            lsmax = -1 * F.log_softmax(all_sims, dim=-1)
            loss = lsmax[-1]
            return loss

        def contrastiveLoss(self, pos_score, neg_scores):
                neg_scores_sub = torch.stack(list(map(self.subMargin, neg_scores)))
                all_scores = torch.cat((neg_scores_sub, pos_score), dim=-1)
                lsmax = -1 * F.log_softmax(all_scores, dim=-1)
                pos_loss = lsmax[-1]
                return pos_loss

        
                
