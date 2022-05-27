import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np
import copy

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

class Attention(nn.Module):
	def __init__(self, vis):
		super(Attention, self).__init__()
		self.vis = vis
		self.num_attention_heads = 8
		self.hidden_size = 1024
		self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(self.hidden_size, self.all_head_size)
		self.key = nn.Linear(self.hidden_size, self.all_head_size)
		self.value = nn.Linear(self.hidden_size, self.all_head_size)

		self.out = nn.Linear(self.hidden_size, self.hidden_size)
		self.attn_dropout = nn.Dropout(0)
		self.proj_dropout = nn.Dropout(0)

		self.softmax = nn.Softmax(dim=-1)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_probs = self.softmax(attention_scores)
		weights = attention_probs if self.vis else None
		attention_probs = self.attn_dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		attention_output = self.out(context_layer)
		attention_output = self.proj_dropout(attention_output)
		return attention_output, weights

class Mlp(nn.Module):
	def __init__(self):
		super(Mlp, self).__init__()
		self.mlp_size = 4096
		self.hidden_size = 1024
		self.fc1 = nn.Linear(self.hidden_size, self.mlp_size)
		self.fc2 = nn.Linear(self.mlp_size, self.hidden_size)
		self.act_fn = ACT2FN["gelu"]
		self.dropout = nn.Dropout(0)

		self._init_weights()

	def _init_weights(self):
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.normal_(self.fc1.bias, std=1e-6)
		nn.init.normal_(self.fc2.bias, std=1e-6)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act_fn(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.dropout(x)
		return x

class Block(nn.Module):
	def __init__(self, vis):
		super(Block, self).__init__()
		self.hidden_size = 1024
		self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
		self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
		self.ffn = Mlp()
		self.attn = Attention(vis)

	def forward(self, x):
		h = x
		x = self.attention_norm(x)
		x, weights = self.attn(x)
		x = x + h

		h = x
		x = self.ffn_norm(x)
		x = self.ffn(x)
		x = x + h
		return x, weights


class TransEncoder(nn.Module):
	def __init__(self, vis):
		super(TransEncoder, self).__init__()
		self.vis = vis
		self.layer = nn.ModuleList()
		self.hidden_size = 1024
		self.encoder_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
		self.num_layers = 4
		for _ in range(self.num_layers):
			layer = Block(vis)
			self.layer.append(copy.deepcopy(layer))

	def forward(self, hidden_states):
		attn_weights = []
		for layer_block in self.layer:
			hidden_states, weights = layer_block(hidden_states)
			if self.vis:
				attn_weights.append(weights)
		encoded = self.encoder_norm(hidden_states)
		return encoded, attn_weights

class TransUNet(nn.Module):
	def __init__(self):
		super(TransUNet, self).__init__()

		self.hidden_size = 1024

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

		#self.position_embeddings = nn.Parameter(torch.zeros(1, 14 * 14, self.hidden_size))
		self.position_embeddings = self.create_position_encoding(14 * 14, self.hidden_size)

		self.transformer = TransEncoder(True)

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

	def create_position_encoding(self, n, d):
		pe = torch.zeros(n, d)
		for pos in range(n):
			for i in range(0, d, 2):
				pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d)))
				pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d)))

		pe = pe.unsqueeze(0)
		device = torch.device("cuda")
		pe = pe.to(device)
		return pe

	def forward(self, x):
		feats = []
	
		for f in self.encoder_blocks:
			x = f(x)
			feats.append(x)
		feats.pop()

		enc_shape = x.shape

		x = x.flatten(2)
		x = x.transpose(-1, -2)
		x = x + self.position_embeddings

		x, _ = self.transformer(x)

		x = x.transpose(-1, -2)
		x = x.view(enc_shape)


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
		#print("output shape", output.shape)
		return output

	def dice_loss(self, X, Y):
		pred = self.forward(X)
		smooth = 1.
		y_true_f = torch.reshape(Y, (-1,))
		y_pred_f = torch.reshape(pred, (-1,))
		intersection = torch.sum(y_true_f * y_pred_f)
		score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
		return (1-score)

	def cal_loss(self, x, g):
		p = self.forward(x)
		p = p.view(p.shape[0], -1)
		max_p = torch.max(p, dim=1, keepdim=True).values
		logloss = nn.BCELoss()
		#print(max_p.shape, g.shape)
		loss = logloss(max_p, g) 
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