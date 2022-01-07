import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np

class DSNet(nn.Module):
	def __init__(self):
		super(DSNet, self).__init__()

		self.encoder_blocks = nn.ModuleList([
			nn.Sequential(Conv2d(1, 64, kernel_size=3),
			Conv2d(64, 64, kernel_size=3),
			nn.MaxPool2d(2, stride=2),
			),
			
			nn.Sequential(Conv2d(64, 128, kernel_size=3),
			Conv2d(128, 128, kernel_size=3),
			nn.MaxPool2d(2, stride=2),
			),

			nn.Sequential(Conv2d(128, 256, kernel_size=3),
			Conv2d(256, 256, kernel_size=3),
			nn.MaxPool2d(2, stride=2),
			Conv2d(256, 512, kernel_size=3)
			),
		])

		self.embedding = Conv2d(512, 1, kernel_size=1, stride=1, padding="same", bias=False)

		#self.softmax = torch.nn.Softmax(2)

	def forward(self, x):
		for f in self.encoder_blocks:
			x = f(x)

		embs = self.embedding(x)
		#print(embs.shape)
		alpha = F.softmax(embs.view(embs.size()[0], 1, -1), dim=2).view_as(embs)
		#print(alpha.shape)
		Mul = torch.mul(x, alpha)
		#print(Mul.shape)
		y = torch.sum(Mul, dim=(2,3))
		#y = F.relu(y)
		#print(y.shape)
		#print(y)
		return alpha, y

	def cal_loss(self, x1, x2, device):
		_, p1 = self.forward(x1)
		_, p2 = self.forward(x2)
		logloss = nn.BCELoss()

		d = F.cosine_similarity(p1, p2, dim=1)
		#print(d)
		y = torch.zeros(d.shape)
		y = y.to(device)
		loss = logloss(d, y) 
		return loss
