import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np
from torchmetrics import Accuracy

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
			nn.Sequential(Conv2d(1024, 512, kernel_size=3),        #1024, 512
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

		self.pairloss_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
		self.pairloss_layer.weight = nn.parameter.Parameter(data=torch.FloatTensor(self.pair_weights(64)), requires_grad=False)
		#self.pairloss_layer.requires_grad_(False)

	def pair_weights(self, dim):
		f = np.zeros((8, dim ,3,3)) #dim = 1
		f[:,:,1,1] = 1
		c = 0
		for i in range(3):
			f[c, :, 0, i] = -1
			c = c+1
		for i in range(3):
			f[c,:, 2, i] = -1
			c = c+1
		f[c, :, 1, 0] = -1
		c = c + 1
		f[c, :, 1, 2] = -1
		return f

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
		return output, x

	def cal_loss0(self, x, g, device):
		p = self.forward(x)
		p = p.view(p.shape[0], -1)
		max_p = torch.max(p, dim=1, keepdim=True).values
		logloss = nn.BCELoss()
		#print(max_p.shape, g.shape)
		loss = logloss(max_p, g) 
		return loss

	def max_loss(self, p, g, device):
		p = p.view(p.shape[0], -1)
		max_p = torch.max(p, dim=1, keepdim=True).values
		logloss = nn.BCELoss()
		loss = logloss(max_p, g)
		accfunc = Accuracy(threshold=0.5)
		accfunc = accfunc.to(device)
		acc = accfunc(max_p, g.int())
		return loss, acc

	def cam_loss(self, p, g, device):
		#p = p.view(p.shape[0], -1)
		#max_p = torch.max(p, dim=1, keepdim=True).values
		avgpool = nn.AvgPool2d(224, padding=0)
		avg_p = avgpool(p)
		avg_p = self.output_block(avg_p)
		logloss = nn.BCELoss()
		loss = logloss(avg_p, g)
		accfunc = Accuracy(threshold=0.5)
		accfunc = accfunc.to(device)
		acc = accfunc(avg_p, g.int())
		return loss, acc

	def cam_loss(self, p, g, device):
		#p = p.view(p.shape[0], -1)
		#max_p = torch.max(p, dim=1, keepdim=True).values
		avgpool = nn.AvgPool2d(224, padding=0)
		avg_p = avgpool(p)
		avg_p = self.output_block(avg_p)
		avg_p = torch.squeeze(avg_p, dim=-1)
		avg_p = torch.squeeze(avg_p, dim=-1)
		logloss = nn.BCELoss()
		loss = logloss(avg_p, g)
		accfunc = Accuracy(threshold=0.5)
		accfunc = accfunc.to(device)
		acc = accfunc(avg_p, g.int())
		return loss, acc

	def proj_loss(self, p1, p2):
		cosine = nn.CosineSimilarity(dim=-1, eps=1e-08)
		p1_m1 = torch.max(p1, dim=2, keepdim=False).values
		p1_m2 = torch.max(p2, dim=3, keepdim=False).values

		p2_m1 = torch.max(p2, dim=2, keepdim=False).values
		p2_m2 = torch.max(p2, dim=3, keepdim=False).values
		p2_m2 = torch.flip(p2_m2, dims=[-1])

		return torch.mean(1-cosine(p1_m1, p2_m2) + 1 - cosine(p1_m2, p2_m1))

	def pair_loss(self, p):
		p = torch.abs(p)
		loss = torch.sum(p, axis=0)
		loss = torch.mean(loss)
		return loss

    def dice_loss(self, X, Y):
        pred = X
        smooth = 1e-10
        y_true_f = torch.reshape(Y, (-1,))
        y_pred_f = torch.reshape(pred, (-1,))
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return (1-score)


	def cal_loss(self, x1, x2, g, device):
		#p1, x = self.forward(x1)
		x1_f1 = torch.flip(x1, dims=[2])
		x1_f2 = torch.flip(x1, dims=[3]) 
		#p2 = self.forward(x2)

		p1, x = self.forward(x1)
		p1_f1, _ = self.forward(x1_f1)
		p1_f2, _ = self.forward(x1_f2)

		p1_f1 = torch.flip(p1_f1, dims=[2])
		p1_f2 = torch.flip(p1_f1, dims=[3])


		pair_diff1 = self.pairloss_layer(p1)
		pair_diff2 = self.pairloss_layer(p1_f1)
		pair_diff3 = self.pairloss_layer(p1_f2)

		loss_m1, acc = self.max_loss(p1, g, device)
		loss_m2, _ = self.max_loss(p1_f1, g, device)
		loss_m3, _ = self.max_loss(p1_f2, g, device)
		loss_m = (loss_m1 + loss_m2 + loss_m3) / 3 
		#loss_p = self.proj_loss(p1, p2)
		pairloss = self.pair_loss(pair_diff1) + self.pair_loss(pair_diff2) + self.pair_loss(pair_diff3)

		dice_l = self.dice_loss(p1, p1_f1) + self.dice_loss(p1, p1_f2) + self.dice_loss(p1_f1, p1_f2) 

		loss = loss_m + (0.01 * pairloss) + (0.1 * dice_l) 

		return loss, acc


	def cal_score(self, X, Y):
		preds, _ = self.forward(X)
		preds = preds.cpu().detach().numpy()
		Y = Y.cpu().detach().numpy()
		batch_size = preds.shape[0]
		dices = []
		preds = preds / np.max(preds)
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
			#dice_val = np.sum(gt) - np.sum(pred)
			dices.append(dice_val)
		return dices

			for i in range(x1.shape[0]):
				img = x1[i, 0, :, :] * 255
				img = img.astype(int)
				pred = preds[i, 0, :, :]
				pred = pred.astype(int)
				pred = pred / np.max(pred)
				pred = pred > 0.5
				seg = pred * 255