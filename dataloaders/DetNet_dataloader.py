import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
import math
import random

class load_data(torch.utils.data.Dataset):
	def __init__(self,fold,mode=0, per=None, seed_select=None, TRAINING_PATH=None, FOLD_PATH=None):
		random.seed(0)
		self.TRAINING_PATH = TRAINING_PATH
		self.FOLD_PATH = FOLD_PATH

		if mode == 0:
			images_indx = self.get_files(fold,'train', per, seed_select)
		if mode == 1:
			images_indx = self.get_files(fold,'val', per, seed_select)
		if mode == 2:
			images_indx = self.get_files(fold,'test')
		random.seed(0)
		
		#random.shuffle(images_indx)
		self.datalist = images_indx

	def __len__(self):
		return int(len(self.datalist))

	def __getitem__(self, idx):
		s = 224
		fname = self.TRAINING_PATH + str(self.datalist[idx][0]) + '.png'
		img_in = cv2.imread(fname)
		img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
		img_in = cv2.resize(img_in, ( s , s ))

		img =img_in[:,:,np.newaxis]/255.0

		if self.datalist[idx][1] == 1:
			g = torch.FloatTensor([1])
		else:
			g = torch.FloatTensor([0])

		img = np.transpose(img, (2, 0, 1))

		img = torch.FloatTensor(img)

		return (img, g)

	def get_files(self, fold,state,per=None, seed_select=None):
		random.seed(0)
		path = self.FOLD_PATH
		pos=np.genfromtxt(path+"glottis"+'.txt',dtype='str')
		neg=np.genfromtxt(path+"noglottis"+'.txt',dtype='str')
		train_size = 2500
		val_size = neg.shape[0] - train_size
		if state == 'train':
			poslist = [(x, 1) for x in pos[:train_size]]
			neglist = [(x, 0) for x in neg[:train_size]]
		else:
			poslist = [(x, 1) for x in pos[train_size:train_size + val_size]]
			neglist = [(x, 0) for x in neg[train_size:]]
		ret_files = poslist
		ret_files.extend(neglist)
		random.shuffle(ret_files)
		return ret_files