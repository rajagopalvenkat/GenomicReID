from __future__ import print_function, division
import tqdm
import torch
import torchfile
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import OrderedDict
import torch.nn.functional as F
import cv2
from PIL import Image
import sys
from tqdm import tqdm
import pandas as pd


##########################################################################
# 							MODEL DEFINITION							 #
##########################################################################


class VGG_16(nn.Module):
	"""
	Main Class
	"""

	def __init__(self):
		"""
		Constructor
		"""
		super().__init__()
		self.block_size = [2, 2, 3, 3, 3]
		self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
		self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
		self.fc6 = nn.Linear(512 * 7 * 7, 4096)
		self.fc7 = nn.Linear(4096, 4096)
		self.fc8 = nn.Linear(4096, 2622)

	def load_weights(self, path):
		""" Function to load luatorch pretrained
		Args:
			path: path for the luatorch pretrained
		"""
		model = torchfile.load(path)
		counter = 1
		block = 1
		for i, layer in enumerate(model.modules):
			if layer.weight is not None:
				if block <= 5:
					self_layer = getattr(self, "conv_%d_%d" % (block, counter))
					counter += 1
					if counter > self.block_size[block - 1]:
						counter = 1
						block += 1
					self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
					self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
				else:
					self_layer = getattr(self, "fc%d" % (block))
					block += 1
					self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
					self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

	def forward(self, x):
		""" Pytorch forward
		Args:
			x: input image (224x224)
		Returns: class logits
		"""
		x = F.relu(self.conv_1_1(x))
		x = F.relu(self.conv_1_2(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv_2_1(x))
		x = F.relu(self.conv_2_2(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv_3_1(x))
		x = F.relu(self.conv_3_2(x))
		x = F.relu(self.conv_3_3(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv_4_1(x))
		x = F.relu(self.conv_4_2(x))
		x = F.relu(self.conv_4_3(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv_5_1(x))
		x = F.relu(self.conv_5_2(x))
		x = F.relu(self.conv_5_3(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc6(x))
		x = F.dropout(x, 0.5, self.training)
		x = F.relu(self.fc7(x))
		x = F.dropout(x, 0.5, self.training)
		return self.fc8(x)


#####################
#     Data Load		#
#####################

skin_dir = 'skin'
sex_dir = 'sex'
eyecolor_dir = 'eyecolor'
hcolor_dir = 'hcolor'

TRAIN = 'train'
VAL='val'
TEST = 'test'

data_transforms = {
	TRAIN: transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
	]),
	VAL: transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
	]),
	TEST: transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
	]),
	TEST_BAK: transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
	]),
	HELDOUT: transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		#transforms.Normalize(mean = [129.1863, 104.7624, 93.5940])
	])
}

skin_datasets = { x: datasets.ImageFolder(os.path.join(skin_dir, x), transform=data_transforms[x]) for x in [TRAIN, TEST, VAL] }
sex_datasets = { x: datasets.ImageFolder(os.path.join(sex_dir, x), transform=data_transforms[x]) for x in [TRAIN, TEST, VAL] }
eyecolor_datasets = { x: datasets.ImageFolder(os.path.join(eyecolor_dir, x), transform=data_transforms[x]) for x in [TRAIN, TEST, VAL] }
hcolor_datasets = { x: datasets.ImageFolder(os.path.join(hcolor_dir, x), transform=data_transforms[x]) for x in [TRAIN, TEST, VAL] }

skin_dataloaders = { x: torch.utils.data.DataLoader(skin_datasets[x], batch_size=8, shuffle=False, num_workers=16)  for x in [TRAIN, TEST, VAL]  }
sex_dataloaders = { x: torch.utils.data.DataLoader(sex_datasets[x], batch_size=8, shuffle=False, num_workers=16)  for x in [TRAIN, TEST, VAL]  }
eyecolor_dataloaders = { x: torch.utils.data.DataLoader(eyecolor_datasets[x], batch_size=8, shuffle=False, num_workers=16)  for x in [TRAIN, TEST, VAL]  }
hcolor_dataloaders = { x: torch.utils.data.DataLoader(hcolor_datasets[x], batch_size=8, shuffle=False, num_workers=16)  for x in [TRAIN, TEST, VAL]  }

skin_sizes = {x: len(skin_datasets[x]) for x in [TRAIN]}
sex_sizes = {x: len(sex_datasets[x]) for x in [TRAIN]}
eyecolor_sizes = {x: len(eyecolor_datasets[x]) for x in [TRAIN]}
hcolor_sizes = {x: len(hcolor_datasets[x]) for x in [TRAIN]}
	
skin_class_names = skin_datasets[TRAIN].classes
sex_class_names = sex_datasets[TRAIN].classes
eyecolor_class_names = eyecolor_datasets[TRAIN].classes
hcolor_class_names = hcolor_datasets[TRAIN].classes

##########################################################################
# 						TRANSFER LEARNING VGGFACE						 #
##########################################################################

use_gpu = torch.cuda.is_available()
# if use_gpu:
# 	print("Using CUDA")

skin_model = VGG_16()
sex_model = VGG_16()
eyecolor_model = VGG_16()
hcolor_model = VGG_16()

##########################################################################
# 								CUSTOM LAYERS							 #
##########################################################################

for param in skin_model.parameters():
	param.requires_grad=False
num_features = skin_model._modules['fc8'].in_features
skin_model._modules['fc8']=nn.Linear(num_features, 3)

for param in sex_model.parameters():
	param.requires_grad=False
num_features = sex_model._modules['fc8'].in_features
sex_model._modules['fc8']=nn.Linear(num_features, 2)

for param in eyecolor_model.parameters():
	param.requires_grad=False
num_features = eyecolor_model._modules['fc8'].in_features
eyecolor_model._modules['fc8']=nn.Linear(num_features, 3)

for param in hcolor_model.parameters():
	param.requires_grad=False
num_features = hcolor_model._modules['fc8'].in_features
hcolor_model._modules['fc8']=nn.Linear(num_features, 3)

if use_gpu:
	skin_model.cuda() #.cuda() will move everything to the GPU side
	sex_model.cuda()
	eyecolor_model.cuda()
	hcolor_model.cuda()

criterion = nn.CrossEntropyLoss()

##########################################################################
# 								LOAD SAVED								 #
##########################################################################

SKIN_PATH = './skinVGG.pth'
SEX_PATH = './sexVGG.pth'
HCOLOR_PATH = './hcolorVGG.pth'
EYECOLOR_PATH = './eyecolorVGG.pth'
skin_model.load_state_dict(torch.load(SKIN_PATH))
sex_model.load_state_dict(torch.load(SEX_PATH))
hcolor_model.load_state_dict(torch.load(HCOLOR_PATH))
eyecolor_model.load_state_dict(torch.load(EYECOLOR_PATH))


def imLoad(image_name):
	image = Image.open(image_name).convert('RGB')
	image = data_transforms[TEST](image)
	image = image.unsqueeze(0)
	return image.cuda()

# for file in os.listdir('custom_directory/'):
# 	x=imLoad('custom_directory/'+file)
# 	print(file,F.softmax(sex_model(x)))

correct = 0
total = 0
with torch.no_grad():
	for i, data in enumerate(sex_dataloaders[TEST]):
		images, labels = data
		if use_gpu:
			images, labels = Variable(images.cuda()), Variable(labels.cuda())
		outputs = sex_model(images)
		_, predicted = torch.max(outputs.data, 1)
		print(predicted)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))