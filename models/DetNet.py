import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np
import cv2
from skimage.measure import regionprops,label

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

		self.embedding = Conv2d(512, 1, kernel_size=1, stride=1, padding="same", bias=False)

		self.fc1 = nn.Linear(512, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 1)

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

		y = F.relu(self.fc1(y))
		y = F.relu(self.fc2(y))
		y = torch.sigmoid(self.fc3(y))

		return alpha, y

	def cal_loss(self, x, g):
		_, p = self.forward(x)
		logloss = nn.BCELoss()
		loss = logloss(p, g) 
		return loss

	def mask_to_box(self, im):
		#im = im[:, :, 0]

		labels_mask = label(im)
		regions = regionprops(labels_mask)
		regions.sort(key=lambda x: x.area, reverse=True)
		if len(regions) > 1:
			for rg in regions[1:]:
				labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
				
			labels_mask[labels_mask!=0] = 1
			im = labels_mask




		xaxis = np.sum(im,axis=0)
		yaxis = np.sum(im,axis=1)
		xs = np.nonzero(xaxis)[0]
		ys = np.nonzero(yaxis)[0]

		positive_mask = np.zeros((28, 28))

		if ((len(xs)<1) or (len(ys)<1)):
			x0,x1,y0,y1 = 0,0,0,0
		else:
			x0 = xs[0]
			x1 = xs[-1]
			y0 = ys[0]
			y1 = ys[-1]

			positive_mask[y0:y1+1, x0:x1+1] = 1
		print(np.sum(im), np.sum(positive_mask))
		return positive_mask

	def dice_score(self, x, y):
		if (np.sum(x) == 0 and np.sum(y) == 0):
			dice_val = 1
		else:
			dice_val = 2.0 * np.sum(np.multiply(x,y))/(np.sum(x)+np.sum(y)+0.000000000000001)
		return dice_val


	def cal_score(self, X, Y):
		preds, _ = self.forward(X)
		preds = preds.cpu().detach().numpy()
		Y = Y.cpu().detach().numpy()
		batch_size = preds.shape[0]
		dices = []
		preds = preds > (1.0 / (preds.shape[-1] * preds.shape[-2]))
		preds = preds.astype(int)


		for i in range(0, batch_size):
			pred = preds[i, :, :, :]
			gt = Y[i, :, :, :]
			pred = np.transpose(pred, (1, 2, 0))
			gt = np.transpose(gt, (1, 2, 0))
			gt = cv2.resize(gt, (pred.shape[0], pred.shape[1]))
			
			pred = pred[:, :, 0]
			pred = self.mask_to_box(pred)
			gt = self.mask_to_box(gt)

			dice_val = self.dice_score(pred, gt)
			dices.append(dice_val)
		return dices
