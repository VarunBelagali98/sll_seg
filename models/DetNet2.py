import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np

'''
Model to detect glottis presence, masks generated from np(seg) == 0 
'''

class DetNet(nn.Module):
	def __init__(self):
		super(DetNet, self).__init__()

		self.encoder_blocks = nn.ModuleList([
			#nn.BatchNorm2d(1),
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

		self.embeddings = nn.ModuleList([
			Conv2d(64, 1, kernel_size=1, stride=1, padding="same", bias=False),
			Conv2d(128, 1, kernel_size=1, stride=1, padding="same", bias=False),
			Conv2d(512, 1, kernel_size=1, stride=1, padding="same", bias=False)
		])


		self.proj = nn.Linear(704, 512)
		self.fc1 = nn.Linear(512, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 1)

		#self.softmax = torch.nn.Softmax(2)

	def forward(self, x):
		enc = []
		for f in self.encoder_blocks:
			x = f(x)
			enc.append(x)

		reps = []
		alphas = []

		for i in range(0, len(enc)):
			embs = self.embeddings[i](enc[i])
			alpha = F.softmax(embs.view(embs.size()[0], 1, -1), dim=2).view_as(embs)
			Mul = torch.mul(enc[i], alpha)
			y = torch.sum(Mul, dim=(2,3))
			reps.append(y)
			alphas.append(alpha)

	
		final_rep = torch.cat(reps, dim=1)

		y = F.relu(self.proj(final_rep))
		y = F.relu(self.fc1(y))
		y = F.relu(self.fc2(y))
		y = torch.sigmoid(self.fc3(y))

		return alphas, y

	def cal_loss(self, x, g):
		_, p = self.forward(x)
		logloss = nn.BCELoss()
		loss = logloss(p, g) 
		return loss
