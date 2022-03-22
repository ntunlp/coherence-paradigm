import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel, XLNetLMHeadModel, XLNetTokenizer


class MIM(nn.Module):
	def __init__(self, model, negs, batch_size, margin, device):
		super().__init__()
		self.batch_size = batch_size
		self.device = device
		
		self.model = XLNetModel.from_pretrained('xlnet-{}-cased'.format(model))
		if model=='base':
			hidden_size = 768
		elif model=='large':
			hidden_size = 1024

		self.negs = negs
		self.margin = margin
		self.getTranspose = lambda x: torch.transpose(x, -2, -1)
		self.subMargin = lambda z: z - margin

		self.conlinear = nn.Linear(hidden_size, 1)
	
	def getScore(self, doc):
		rep = self.model(input_ids=doc).last_hidden_state[:,-1,:]
		score = self.conlinear(rep).view(-1)
		return score
		
	def forward(self, pos_doc, neg_docs):
		pos_score = self.getScore(pos_doc)
		neg_scores = list(map(self.getScore, list(neg_docs)))
		return pos_score, neg_scores[0]
	
	def pairwiseLoss(self, pos_score, neg_score):
		zero_tensor = torch.zeros_like(pos_score)
		margin_tensor = torch.tensor(self.margin).cuda()
		loss_tensor = margin_tensor + neg_score - pos_score
		loss = torch.max(zero_tensor, loss_tensor)

		return loss






		
