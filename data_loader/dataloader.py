from torchvision import transforms
import torch
import random
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# define data class for 1D CNN reading from csv
class TimeSeriesDataset:
	
	def __init__(self,data,data_path,padding=True,training=True,normalize=True):
		self.data_path = data_path
		self.data = data
		self.training = training
		self.padding = padding
	
	def __getitem__(self,idx):
		# read data information
		data_path,abnormal,lead1,lead3,hr,severity = self.data.iloc[idx,:].values.tolist()
		# data = np.array(pd.read_csv(open(data_path,'r')), dtype = np.float).reshape(-1)
		data_3 = self.read_lead(lead3,"III")
		data_2 = self.read_lead(data_path,"II")
		# normalize
		# data = self.std_normalize(data)
		data_3 = self.std_normalize(data_3)
		data_2 = self.std_normalize(data_2)
		# padding if on
		# if (self.padding):
		# 	left = int(data.shape[0] * random.uniform(0,0.15))
		# 	right = int(data.shape[0] * random.uniform(0,0.15))
		# 	# data = np.pad(data,(left,right),'constant',constant_values=(0,0))
		
		# concat severity
		if(int(severity)==3):
				severity=2
		hr = int(hr)
		abnormal = int(abnormal)

		hr_tensor=torch.FloatTensor([hr])
		abnormal_tensor= torch.LongTensor([int(abnormal)])
		severity_tensor = torch.LongTensor([int(severity)])
		# choose which label to use
		label_tensor = abnormal_tensor
		data_tensor = torch.FloatTensor([data_2])
		data_3_tensor = torch.FloatTensor([data_3])

		return data_tensor,data_3_tensor,label_tensor

	def __len__(self):
		return len(self.data)
	
	def std_normalize(self, data):
		mean = np.mean(data)
		std = np.math.sqrt(np.mean(np.power((data - mean), 2)))
		return (data - mean) / std

	def read_lead(self,data_path,lead_num="III"):
		if ("VT" in data_path):
			data_path = os.path.join( "/data/Vien_Tim/D123_Short/"+"D"+lead_num+"_Short",data_path)
		else:
			data_path = os.path.join( "/data/Viet_Gia_Clinic/D123/"+"D"+lead_num,data_path)
		data = np.array(pd.read_csv(open(data_path,'r')), dtype = np.float).reshape(-1)
		return data
# define a data class
class ClassificationDataset:
	def __init__(self, data, data_path, transform, training=True):
		"""Define the dataset for classification problems

		Args:
			data ([dataframe]): [a dataframe that contain 2 columns: image name and label]
			data_path ([str]): [path/to/folder that contains image file]
			transform : [augmentation methods and transformation of images]
			training (bool, optional): []. Defaults to True.
		"""
		self.data = data
		self.imgs = data["file_name"].unique().tolist()
		self.data_path = data_path
		self.training = training
		self.transform = transform

	def __getitem__(self, idx):
		img = Image.open(os.path.join(self.data_path, self.data.iloc[idx, 0]))
		img = img.convert('RGB')
		label = self.data.iloc[idx, 1]
		label = [int(x) for x in label.split(' ')]
		label = torch.tensor(label, dtype=torch.int8)

		if self.transform is not None:
			img = self.transform(img)
		
		return img, label

	def __len__(self):
		return len(self.imgs)


def make_loader(dataset, train_batch_size, validation_split=0.2):
	"""make dataloader for pytorch training

	Args:
		dataset ([object]): [the dataset object]
		train_batch_size ([int]): [training batch size]
		validation_split (float, optional): [validation ratio]. Defaults to 0.2.

	Returns:
		[type]: [description]
	"""
	# number of samples in train and test set
	train_len = int(len(dataset) * (1 - validation_split))
	test_len = len(dataset) - train_len
	train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
	# create train_loader
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=train_batch_size, shuffle=True,
	)
	# create test_loader
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,)
	return train_loader, test_loader


def data_split(data, test_size):
	x_train, x_test, y_train, y_test = train_test_split(
		data, data["label"], test_size=test_size
	)
	return x_train, x_test, y_train, y_test
