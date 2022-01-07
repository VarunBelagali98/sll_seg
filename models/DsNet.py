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
			Conv2d(128, 128, kernel_size=3)
            nn.MaxPool2d(2, stride=2),
            ),

			nn.Sequential(Conv2d(128, 256, kernel_size=3),
			Conv2d(256, 256, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            ),

			nn.Sequential(Conv2d(256, 512, kernel_size=3),
			Conv2d(512, 512, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            Conv2d(512, 512, kernel_size=1),
            ),
		])

        self.embedding = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding="same", bias=False)

        #self.softmax = torch.nn.Softmax(2)

	def forward(self, x):
		x = x[0, :, :, :, :]
        for f in self.encoder_blocks:
			x = f(x)

        embs = self.embedding(x)
        alpha = F.softmax(embs.view(em.size()[0], 1, -1)).view_as(embs)
        Mul = torch.mul(em, alpha)
        y = torch.sum(Mul, dim=(2,3))
		return y

	def cal_loss(self, X):
		pred = self.forward(X)
        logloss = nn.BCELoss()

        p1 = pred[0,:]
        p2 = pred[1,:]
        d = F.cosine_similarity(p1, p2, dim=0)
        y = torch.ones(0)
        return logloss(d.unsqueeze(0), y)
