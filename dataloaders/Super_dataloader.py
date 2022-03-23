import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
import math

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
		
		#random.shuffle(images_indx)
		self.datalist = images_indx

	def __len__(self):
		return int(len(self.datalist))

	def __getitem__(self, idx):
		s = 224
		fname = self.TRAINING_PATH + str(self.datalist[idx]) + '.png'
		img_in = cv2.imread(fname)
		img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
		img_in = cv2.resize(img_in, ( s , s ))

		fname = self.TRAINING_PATH + str(self.datalist[idx]) + '_seg.png'
		img_g = cv2.imread(fname)
		img_g = cv2.resize(img_g, ( s , s )) 
			
		im = img_g[:,:,0]/255
		xaxis = np.sum(im,axis=0)
		yaxis = np.sum(im,axis=1)
		xs = np.nonzero(xaxis)[0]
		ys = np.nonzero(yaxis)[0]

		negative_mask = np.ones((s, s, 1))
		positive_mask = np.zeros((s, s, 1))

		if ((len(xs)<1) or (len(ys)<1)):
			x0,x1,y0,y1 = 0,0,0,0
		else:
			x0 = xs[0]
			x1 = xs[-1]
			y0 = ys[0]
			y1 = ys[-1]
	
			positive_mask[y0:y1+1, x0:x1+1, :] = 1
			negative_mask[y0:y1+1, x0:x1+1, :] = 0
			
		img =img_in[:,:,np.newaxis]/255.0
		#pos = positive_mask[:,:,:]
		#neg = negative_mask[:,:,:]
		img_ann = img_g[:,:,1,np.newaxis]/255.0

		img = np.transpose(img, (2, 0, 1))
		img_ann = np.transpose(img_ann, (2, 0, 1))

		img = torch.FloatTensor(img)
		img_ann = torch.FloatTensor(img_ann)

		return (img, img_ann)

	def get_files(self, fold,state,per=None, seed_select=None):
		random.seed(0)
		train_list = [1, 2, 3, 4, 5]
		test = fold
		validate = (fold+1)%6
		if validate == 0:
			validate = 1
		train_list.remove(test)
		train_list.remove(validate)
		print('..test',test)
		path = self.FOLD_PATH
		if state == 'train':
			files1=np.genfromtxt(path+str(train_list[2])+'.txt',dtype='str')
			files2=np.genfromtxt(path+str(train_list[1])+'.txt',dtype='str')
			files3=np.genfromtxt(path+str(train_list[0])+'.txt',dtype='str')
			ret_file = np.append(files1,files2)
			ret_file = np.append(ret_file,files3)
		if state == 'val':
			ret_file = np.genfromtxt(path+str(validate)+'.txt',dtype='str')
		if state == 'test':
			ret_file = np.genfromtxt(path+str(test)+'.txt',dtype='str')
			print('test mode', test, ret_file.shape[0])
		if per != None:
			if seed_select != None:
				print("subset selection seed ", seed_select)
				random.seed(seed_select)
				ret_file = random.sample(list(ret_file), int(ret_file.shape[0]*per/100))
				random.seed(0)
			else:
				ret_file = random.sample(list(ret_file), int(ret_file.shape[0]*per/100))
		print(state, len(ret_file))
		return list(ret_file)