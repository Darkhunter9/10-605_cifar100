# -*- coding: utf-8 -*-
"""10-605_Project_CIFAR-100-RandomForests.ipynb

Automatically generated by Colaboratory.

Original file is located at
	https://colab.research.google.com/drive/160WKPi-wxhyeFqRLjdCEHIqFZWZGA3Rz

# **This is the Jupyter / Google Collab notebook for the RandomForest part of the 10-605 Project Fall 2020 by the Team Zihao Ding, Varun Rawal, Sharan Sheshadri**





# NEXT : CIFAR-100 WORK

# CIFAR 100 Image Classification 

 This dataset contains 100 different classes of image. Each classes contain 500 other images therefore we can say the data is properly organised.  All images are of `3 channels` of dimensions `32 x 32` . We will be applying different Random Forest classifier approaches to get the best outputs from this dataset. 
 
 I would like to mention [This site](https://www.kaggle.com/minbavel/cifar-100-images) from where I took the dataset on which I will be working on .
"""

#!pip install opendatasets --upgrade -q
#!pip install jovian --upgrade -q

# Commented out IPython magic to ensure Python compatibility.
#@title Import modules for pytorch code
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.models as models
import matplotlib.pyplot as plt
import opendatasets as od
# %matplotlib inline

kaggle_api = {"username":"varunrawal","key":"16c2afe690b4b18e912ad53dc7424900"}

# dataset_url = 'https://www.kaggle.com/minbavel/cifar-100-images'
# od.download(dataset_url)

"""## Preparing the Data

Let's begin by downloading the dataset and creating PyTorch datasets to load the data.

Here in my Project I will be using dataset that is already present in Kaggle data section. 
I am using CIFAR 100 dataset from https://www.kaggle.com/minbavel/cifar-100-images
"""

project_name="Cifar-100"

# I dowloaded the dataset using OpenDataset Library and by entering my Kaggle Key and Kaggle Username
# Let's look into the data directory
data_dir = './cifar-100-images/CIFAR100'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/TRAIN")
print(classes[:10])
print(len(classes))

# Let's evaluate a single class say "man"
man_file=os.listdir(data_dir+"/TRAIN/man")
print("NO. of Training examples for Man:",len(man_file))
print(man_file[:5])

# Let's see how many number of files/images are present in each classes
di={}
for i in classes:
	di[i]=len(os.listdir(data_dir+"/TRAIN/"+i))
print(di)

"""# Training the Dataset """

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4,padding_mode='reflect'), 
						 tt.RandomHorizontalFlip(), 
						 tt.ToTensor(), 
						 tt.Normalize(*stats,inplace=True)
						])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)
						])
# PyTorch datasets
train_ds = ImageFolder(data_dir+'/TRAIN', train_tfms)
valid_ds = ImageFolder(data_dir+'/TEST', valid_tfms)

# PyTorch data loaders
train_dl = DataLoader(train_ds, len(train_ds), shuffle=True, num_workers=0, pin_memory=True)
valid_dl = DataLoader(valid_ds, len(valid_ds), num_workers=0, pin_memory=True)

def show_batch(dl):
	for images, labels in dl:
		fig, ax = plt.subplots(figsize=(12, 12))
		ax.set_xticks([]); ax.set_yticks([])
		ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
		break

"""Using a GPU
To seamlessly use a GPU, if one is available, we define a couple of helper functions (get_default_device & to_device) and a helper class DeviceDataLoader to move our model & data to the GPU as required.
"""

#@title Device Data loader modules for pytorch code

def get_default_device():
	"""Pick GPU if available, else CPU"""
	if torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')
	
def to_device(data, device):
	"""Move tensor(s) to chosen device"""
	if isinstance(data, (list,tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking=True)

class DeviceDataLoader():
	"""Wrap a dataloader to move data to a device"""
	def __init__(self, dl, device):
		self.dl = dl
		self.device = device
		
	def __iter__(self):
		"""Yield a batch of data after moving it to device"""
		for b in self.dl: 
			yield to_device(b, self.device)

	def __len__(self):
		"""Number of batches"""
		return len(self.dl)

device = get_default_device()
print("device : ", device)

# Transfering data to the device in use (In our case GPU)
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

from sklearn.metrics import log_loss
import pickle 
import os

from sklearn.preprocessing import OneHotEncoder

def cross_entropy(predictions, targets):
	N = predictions.shape[0]
	ce = -np.sum(targets * np.log(predictions)) / N
	return ce

def compute_loss_from_prob(model, images, labels):

  y_pred = model.predict_proba(images)
  y_true = labels

  y_true = OneHotEncoder().fit_transform(labels.reshape(-1,1)).toarray()

  print(y_pred.shape, y_true.shape)
  assert(y_pred.shape == y_true.shape)

  #ce_loss = cross_entropy(y_pred, y_true)
  ll_loss = log_loss(y_true, y_pred)

  print(ce_loss, ll_loss)
  #assert(ce_loss == ll_loss)
  return ll_loss

def predicts(self, vectors):

  n = vectors.shape[0]
  predictions = np.zeros((n,))

  for i in range(n):
	  vector = vectors[i, :].reshape(1, -1)
	  #print(vector.shape)
	  predictions[i] = self.predict(vector)

  return predictions

def accuracy(outputs, labels):
	#_, preds = torch.max(outputs, dim=1)
	preds = outputs
	return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def training_step(self, batch, param_identifier):

	images, labels = batch 
	images = images.cpu()
	labels = labels.cpu()
	images = images.reshape(images.shape[0], -1)

	filename = param_identifier

	if os.path.exists(filename):
		with open(filename, "rb") as file:
			self = pickle.load(file)
	else:
		#print(images.shape)
		self.fit(images, labels)
		#out = predicts(self, images)          # Generate predictions
		#loss = F.mse_loss(out, labels) # Calculate loss

	train_acc = self.score(images, labels) # Calculate acc

	train_loss = compute_loss_from_prob(self, images, labels)

	print("Train Accuracy : ", 100.0*train_acc)
	print("Train Loss : ", train_loss)

	with open(filename, "wb") as file:
	  pickle.dump(self, file)

	return train_acc, train_loss, self

def validation_step(self, batch):
	images, labels = batch 

	images = images.cpu()
	labels = labels.cpu()

	images = images.reshape(images.shape[0], -1)

	#out = predicts(self, images)            # Generate predictions
	#print("@@@ : ", out.shape, labels.shape)
	#loss = F.mse_loss(out, labels)   # Calculate loss

	val_acc = self.score(images, labels) # Calculate acc

	val_loss = compute_loss_from_prob(self, images, labels)

	return {'val_acc': val_acc, 'val_loss' : val_loss}
	
def validation_epoch_end(self, outputs):

	batch_losses = [x['val_loss'] for x in outputs]
	epoch_loss = np.stack(batch_losses).mean()   # Combine losses

	batch_accs = [x['val_acc'] for x in outputs]
	epoch_acc = np.stack(batch_accs).mean()      # Combine accuracies

	return {'val_acc': epoch_acc, 'val_loss': epoch_loss}

def epoch_end(self, epoch, result):
	print("Epoch [{}], val_acc: {:.4f}".format(
		epoch, result['val_acc']))

"""# load setup.py and requirements.txt from https://github.com/ValentinFigue/Sklearn_PyTorch"""

# Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# auth_code = "4/1AY0e-g77YU5GrrEIuY9pKKvhkbcBy4mGHQn5RiiuubgH3jtfBcUCh_gf4M4"
# drive.mount('/content/drive/')

#drive.mount('/content/drive/MyDrive/Sklearn_PyTorch-master')
# %cd /content/drive/MyDrive/Sklearn_PyTorch-master/

# %run -i /content/drive/MyDrive/Sklearn_PyTorch-master/setup.py install
# import sys
# !{sys.executable} -m pip install -r requirements.txt

# Commented out IPython magic to ensure Python compatibility.

# %cd ./
# %pwd

# # verify if sklearn pytorch library impported
# import Sklearn_PyTorch

# # Commented out IPython magic to ensure Python compatibility.
# # %run -i setup.py install

# # Import of the model
# from Sklearn_PyTorch import TorchRandomForestClassifier

# # Initialisation of the model
# my_RF_model = TorchRandomForestClassifier(nb_trees=1, nb_samples=2, max_depth=5, bootstrap=True)

# # Definition of the input data
# import torch
# my_data = torch.FloatTensor([[0,1,2.4],[1,2,1],[4,2,0.2],[8,3,0.4], [4,1,0.4]])
# my_label = torch.LongTensor([0,1,0,0,1])

# print(my_data.shape, my_label.shape)

# # Fitting function
# my_RF_model.fit(my_data, my_label)

# # Prediction function
# my_vector = torch.FloatTensor([[1,2,1.4], [3,3,3]])
# my_result = predicts(my_RF_model, my_vector)
# my_result

#my_RF_model = TorchRandomForestClassifier(nb_trees=1000, nb_samples=30, max_depth=5, bootstrap=True)

"""# Train the Model"""

@torch.no_grad()
def evaluate(model, val_loader):
	#model.eval()
	outputs = [validation_step(model, batch) for batch in val_loader]
	return validation_epoch_end(model, outputs)

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def fit_one_cycle(n_estimators, epochs, max_lr, model, train_loader, val_loader, 
				  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, parameter_identifier = ""):
	torch.cuda.empty_cache()
	history = []
	
	# # Set up cutom optimizer with weight decay
	# optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
	# # Set up one-cycle learning rate scheduler
	# sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
	#                                             steps_per_epoch=len(train_loader))
	epochs = 1
	for epoch in range(epochs):
		# Training Phase 
		#model.train()
		train_losses = []
		train_acc = []

		lrs = []
		for batch in train_loader:

			acc, loss, model = training_step(model,batch, parameter_identifier)

			train_losses.append(loss)
			train_acc.append(acc)
			# loss.backward()
			
			# Gradient clipping
			# if grad_clip: 
			#     nn.utils.clip_grad_value_(model.parameters(), grad_clip)
			
			# optimizer.step()
			# optimizer.zero_grad()
			
			# Record & update learning rate
			#lrs.append(get_lr(optimizer))
			#sched.step()
		
		# Validation phase
		result = evaluate(model, val_loader)

		result['train_loss'] = np.stack(train_losses).mean()
		result['train_acc'] = np.stack(train_acc).mean()

		#result['lrs'] = lrs
		epoch_end(model, epoch, result)
		result['n_estimators'] = n_estimators
		history.append(result)
	return history

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/

batches = [batch for batch in  valid_dl]
imgs, lbls = batches[0]
imgs.shape

# history = [evaluate(my_RF_model, valid_dl)]
# history

"""# Set the hyper-parameters"""

epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay =1e-4
opt_func = torch.optim.Adam



# Plotted the accuracy Graph
def plot_accuracies(history):

	fig= plt.figure(figsize=(15, 5))

	val_accuracies = [100.0*x['val_acc'] for x in history]
	train_accuracies = [100.0*x['train_acc'] for x in history]
	
	n_ests = [x['n_estimators'] for x in history]


	plt.subplot(1, 2, 1)
	plt.plot(n_ests, val_accuracies, '-bx')
	plt.legend(['Validation'])

	plt.subplot(1, 2, 2)
	plt.plot(n_ests, train_accuracies, '-rx')
	plt.legend(['Training'])


	plt.tight_layout()
	plt.xlabel('n_trees')
	plt.ylabel('accuracy %')
	plt.title('Accuracy vs. No. of Trees')

	plt.savefig("RF Accuracies v_s Num_Trees Part 2.png", bbox_inches='tight')


# Training and Validation loss graph
def plot_losses(history):

	plt.clf()

	train_losses = [x.get('train_loss') for x in history]
	val_losses = [x['val_loss'] for x in history]

	n_ests = [x['n_estimators'] for x in history]

	plt.plot(n_ests, train_losses, '-bx')
	plt.plot(n_ests, val_losses, '-rx')
	
	plt.xlabel('epoch')
	plt.ylabel('loss')

	plt.legend(['Training', 'Validation'])

	plt.title('Loss vs. No. of Trees')

	plt.savefig("RF Losses v_s Num_Trees Part 2.png", bbox_inches='tight')



# Commented out IPython magic to ensure Python compatibility.
# %%time
from sklearn.ensemble import RandomForestClassifier

#model = my_RF_model
history = []
n_estimators = 1
my_RF_model = RandomForestClassifier(n_estimators= n_estimators, criterion='gini', max_depth= max(50, n_estimators/10), min_samples_split=30, n_jobs = None, warm_start=True)

for n_estimators in [1, 10, 25, 35, 40, 50, 100, 200, 500, 1000, 5000]:
#for n_estimators in [1, 5]:

	parameter_identifier = f"RF_P2_n_estimators={my_RF_model.n_estimators}, max_depth= {my_RF_model.max_depth}.rfmodel"
	#my_RF_model = TorchRandomForestClassifier(nb_trees = n_estimators, nb_samples=30, max_depth=max(5, n_estimators), bootstrap=True)
	my_RF_model.n_estimators = n_estimators

	history += fit_one_cycle(n_estimators, epochs, max_lr, my_RF_model, train_dl, valid_dl, 
							  grad_clip=grad_clip, 
							  weight_decay=weight_decay, 
							  opt_func=opt_func, parameter_identifier = parameter_identifier)

	plot_accuracies(history)
	plot_losses(history)

plot_accuracies(history)
plot_losses(history)

"""# **Prediction and testing of MODEL**"""

from torchvision.transforms import ToTensor
test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)
						])
test_dataset = ImageFolder(data_dir+'/TEST', test_tfms)
dataset = ImageFolder(data_dir+'/TRAIN', transform=ToTensor())

def predict_image(img, model):
	# Convert to a batch of 1

	#xb = to_device(img.unsqueeze(0), device)
	xb = img.unsqueeze(0).cpu()
	xb = xb.reshape(xb.shape[0], -1)
	# Get predictions from model
	yb = model.predict(xb)
	# Pick index with highest probability

	#_, preds  = np.max(yb)
	preds = yb

	# Retrieve the class label
	return dataset.classes[preds[0]]

# model = my_RF_model
# img, label = test_dataset[550]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# img, label = test_dataset[16]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# img, label = test_dataset[117]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# img, label = test_dataset[210]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# img, label = test_dataset[589]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
# result = evaluate(model, test_loader)
# result

"""## Save and Commit

Let's save the weights of the model, record the hyperparameters, and commit our experiment to Jovian. As you try different ideas, make sure to record every experiment so you can look back and analyze the results.
"""

# torch.save(model.state_dict(), 'cifar100-resnet12layers.pth')
# model2 = to_device(ResNet152(3, 100), device)
# model2.load_state_dict(torch.load('cifar100-resnet12layers.pth'))
# evaluate(model2, test_loader)
