import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np

class ReNet(nn.Module):
	def __init__(self):
		super(ReNet, self).__init__()

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

		self.seg_block = nn.Sequential(nn.Conv2d(64, 5, kernel_size=3, stride=1, padding="same"),
			nn.BatchNorm2d(5),
			nn.Softmax(dim=1))

		self.rec_encoder_blocks = nn.ModuleList([
			nn.Sequential(Conv2d(6, 64, kernel_size=3),
			Conv2d(64, 64, kernel_size=3),),
			
			nn.Sequential(nn.MaxPool2d(2, stride=2),
			Conv2d(64, 128, kernel_size=3),
			Conv2d(128, 128, kernel_size=3),),

			nn.Sequential(nn.MaxPool2d(2, stride=2),
			Conv2d(128, 256, kernel_size=3)),
		])


		self.rec_decoder_blocks = nn.ModuleList([
			nn.Sequential(Conv2d(256, 128, kernel_size=3),
			nn.Upsample( scale_factor=(2,2))
			),
			nn.Sequential(Conv2d(128, 128, kernel_size=3),
			Conv2d(128, 128, kernel_size=3, residual=True),
			Conv2d(128, 64, kernel_size=3),
			nn.Upsample(scale_factor=(2,2))#, scale_factor=(2,2))
			),

			nn.Sequential(Conv2d(64, 64, kernel_size=3),
			Conv2d(64, 64, kernel_size=3, residual=True),
			Conv2d(64, 64, kernel_size=3),
			)
			])

		self.output_block = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding="same"),
			nn.BatchNorm2d(1),
			nn.Sigmoid())

	def forward(self, x, xc):
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
		
		seg_out = self.seg_block(x)

		x1 = torch.cat((seg_out, xc), dim=1)

		x = x1
		for f in self.rec_encoder_blocks:
			x = f(x)

		for f in self.rec_decoder_blocks:
			x = f(x)
		
		rec_out = self.output_block(x)
		
		return seg_out, rec_out

	def cal_loss(self, x, xc):
		_, p = self.forward(x, xc)
		cal_loss = nn.MSELoss()
		loss = cal_loss(p, x)
		return loss