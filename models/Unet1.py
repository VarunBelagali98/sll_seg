import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()

		self.encoder_blocks = nn.ModuleList([
			nn.Sequential(Conv2d(1, 64, kernel_size=3),
			Conv2d(64, 64, kernel_size=3),),
			
			nn.Sequential(nn.MaxPool2d(2, stride=2),
			Conv2d(64, 128, kernel_size=3),
			Conv2d(128, 128, kernel_size=3)),

			nn.Sequential(nn.MaxPool2d(2, stride=2),
			Conv2d(128, 256, kernel_size=3),
			Conv2d(256, 256, kernel_size=3)),

			nn.Sequential(nn.MaxPool2d(2, stride=2),
			Conv2d(256, 512, kernel_size=3),
			Conv2d(512, 512, kernel_size=3)),

			nn.Sequential(nn.MaxPool2d(2, stride=2),
			Conv2d(512, 1024, kernel_size=3)),
		])


		self.decoder_blocks = nn.ModuleList([
			nn.Sequential(Conv2d(1024, 512, kernel_size=3),
			nn.Upsample( scale_factor=(2,2))
			),

			nn.Sequential(Conv2d(1024, 512, kernel_size=3),
			Conv2d(512, 256, kernel_size=3),
			nn.Upsample(scale_factor=(2,2))
			),

			nn.Sequential(Conv2d(512, 256, kernel_size=3),
			Conv2d(256, 128, kernel_size=3),
			nn.Upsample(scale_factor=(2,2))#, scale_factor=(2,2))
			),

			nn.Sequential(Conv2d(256, 128, kernel_size=3),
			Conv2d(128, 64, kernel_size=3),
			nn.Upsample(scale_factor=(2,2))#, scale_factor=(2,2))
			),

			nn.Sequential(Conv2d(128, 64, kernel_size=3),
			Conv2d(64, 64, kernel_size=3),
			)
			])

		self.output_block = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding="same"),
			nn.BatchNorm2d(1),
			nn.Sigmoid()) 

	def forward(self, x):
		feats = []
	
		for f in self.encoder_blocks:
			x = f(x)
			feats.append(x)
		feats.pop()
		for f in self.decoder_blocks:
			x = f(x)
			try:
				if len(feats) > 0:
					x = torch.cat((x, feats[-1]), dim=1)
					feats.pop()
			except Exception as e:
				print(x.size())
				print(feats[-1].size())
				raise e


		output = self.output_block(x)
		return output

	def loss_max(self, p, device):
		logloss = nn.BCELoss()
		p = p.view(p.shape[0], -1)
		max_p = torch.max(p, dim=1, keepdim=True).values
		g = torch.ones(max_p.shape)
		g = g.to(device)
		loss = logloss(max_p, g)
		return loss

	def loss_bright(self, p, x, device):
		logloss = nn.BCELoss()
		mask = (x > 0.2).float()
		g = torch.zeros(p.shape)
		g = g.to(device)
		pm = torch.mul(p, mask)
		gm = torch.mul(g, mask)
		loss = logloss(pm, gm)
		return loss

	def proj_loss(self, p1, p2):
		cosine = nn.CosineSimilarity(dim=-1, eps=1e-08)
		p1_m1 = torch.max(p1, dim=2, keepdim=False).values
		p1_m2 = torch.max(p2, dim=3, keepdim=False).values

		p2_m1 = torch.max(p2, dim=2, keepdim=False).values
		p2_m2 = torch.max(p2, dim=3, keepdim=False).values
		p2_m2 = torch.flip(p2_m2, dims=[-1])

		return torch.mean(1-cosine(p1_m1, p2_m2) + 1 - cosine(p1_m2, p2_m1))

	def cal_loss(self, x1, x2, device):
		p1 = self.forward(x1)
		p2 = self.forward(x2)

		loss_m = self.loss_max(p1, device) + self.loss_max(p2, device)
		loss_b = self.loss_bright(p1, x1, device) + self.loss_bright(p2, x2, device)
		loss_p = self.proj_loss(p1, p2)

		loss = loss_m + loss_b + loss_p

		return loss

	def cal_score(self, X, Y):
		preds = self.forward(X)
		preds = preds.cpu().detach().numpy()
		Y = Y.cpu().detach().numpy()
		batch_size = preds.shape[0]
		dices = []

		preds = preds > 0.5
		preds = preds.astype(int)

		for i in range(0, batch_size):
			pred = preds[i, :, :, :]
			gt = Y[i, :, :, :]
			if (np.sum(gt) == 0 and np.sum(pred) == 0):
				dice_val = 1
				iou=1
			else:
				dice_val = 2.0 * np.sum(np.multiply(pred,gt))/(np.sum(pred)+np.sum(gt)+0.000000000000001)
				iou = np.sum(np.multiply(pred,gt))/(np.sum(pred)+np.sum(gt)- np.sum(np.multiply(pred,gt)) +0.000000000000001)
			dices.append(dice_val)
		return dices

	