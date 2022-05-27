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
		fname = self.TRAINING_PATH + str(self.datalist[idx]) + '.png'
		img1 = cv2.imread(fname)
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img1 = cv2.resize(img1, ( s , s ))

		fname = self.TRAINING_PATH + str(self.datalist[idx]) + '.png'  # same image
		img2 = cv2.imread(fname)
		img2 = cv2.rotate(img2, cv2.cv2.ROTATE_90_CLOCKWISE) # rotate
		#img2 = imutils.rotate(img2, random.randint(0, 360)) 
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		img2 = cv2.resize(img2, ( s , s ))
			
		img1 =img1[:,:,np.newaxis]/255.0
		img2 =img2[:,:,np.newaxis]/255.0

		img1 = np.transpose(img1, (2, 0, 1))
		img2 = np.transpose(img2, (2, 0, 1))

		img1 = torch.FloatTensor(img1)
		img2 = torch.FloatTensor(img2)

		return (img1, img2)

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