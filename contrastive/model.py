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
		#set requires_grad=False for parameters in self.model.parameters() to remove fine-tuning for XLNet representations
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

	def contrastiveLoss(self, pos_score, neg_scores):
		neg_scores_sub = torch.stack(list(map(self.subMargin, neg_scores)))
		all_scores = torch.cat((neg_scores_sub, pos_score), dim=-1)

		lsmax = -1 * F.log_softmax(all_scores, dim=-1)

		pos_loss = lsmax[-1]
		
		return pos_loss

