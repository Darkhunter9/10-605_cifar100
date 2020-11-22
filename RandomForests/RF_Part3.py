
"""# Autoencoder (CIFAR-100) + Scikit-Learn Random Forest Classifier

- Runs on CPU or GPU (if available)

A simple, single-hidden-layer, fully-connected autoencoder that compresses 2x 32 x 32 -pixel CIFAR-100 images into 32-pixel vectors (32-times smaller representations). A random forest classifier is then trained for predicting the class labels based on that 32-pixel compressed space.

## Imports
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle

if torch.cuda.is_available():
		torch.backends.cudnn.deterministic = True

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#KMP_DUPLICATE_LIB_OK=TRUE

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 5
batch_size = 256

# Architecture
num_features = 3*32*32
num_hidden_1 = 3*32


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.CIFAR100(root='data', 
															 train=True, 
															 transform=transforms.ToTensor(),
															 download=True)

test_dataset = datasets.CIFAR100(root='data', 
															train=False, 
															transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
													batch_size=len(train_dataset), 
													shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
												 batch_size=len(test_dataset), 
												 shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
		print('Image batch dimensions:', images.shape)
		print('Image label dimensions:', labels.shape)
		break

"""## Model"""

##########################
### MODEL
##########################

class Autoencoder(torch.nn.Module):

		def __init__(self, num_features):
				super(Autoencoder, self).__init__()
				
				### ENCODER
				
				self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
				# The following to lones are not necessary, 
				# but used here to demonstrate how to access the weights
				# and use a different weight initialization.
				# By default, PyTorch uses Xavier/Glorot initialization, which
				# should usually be preferred.
				self.linear_1.weight.detach().normal_(0.0, 0.1)
				self.linear_1.bias.detach().zero_()
				
				### DECODER
				self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
				self.linear_1.weight.detach().normal_(0.0, 0.1)
				self.linear_1.bias.detach().zero_()
				
		def encoder(self, x):
				encoded = self.linear_1(x)
				encoded = F.leaky_relu(encoded)
				return encoded
		
		def decoder(self, encoded_x):
				logits = self.linear_2(encoded_x)
				decoded = torch.sigmoid(logits)
				return decoded
				

		def forward(self, x):
				
				### ENCODER
				encoded = self.encoder(x)
				
				### DECODER
				decoded = self.decoder(encoded)
				
				return decoded

		
torch.manual_seed(random_seed)
model = Autoencoder(num_features=num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""## Training"""

# Commented out IPython magic to ensure Python compatibility.
image_dim = 32 # for CIFAR - 100
#image_dim = 32 # for MNIST
num_channels = 3

start_time = time.time()
for epoch in range(num_epochs):
		for batch_idx, (features, targets) in enumerate(train_loader):
				
				# don't need labels, only the images (features)
				features = features.view(-1, num_channels*image_dim*image_dim).to(device)
						
				### FORWARD AND BACK PROP
				decoded = model(features)
				cost = F.binary_cross_entropy(decoded, features)
				optimizer.zero_grad()
				
				cost.backward()
				
				### UPDATE MODEL PARAMETERS
				optimizer.step()
				
				### LOGGING
				if not batch_idx % 50:
						print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                    %(epoch+1, num_epochs, batch_idx, 
										 len(train_loader), cost))
						
		print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
		
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

"""### Training Dataset"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

##########################
### VISUALIZATION
##########################


train_loader = DataLoader(dataset=train_dataset, 
													batch_size=15, 
													shuffle=True)

# Checking the dataset
for images, labels in train_loader:  
		print('Image batch dimensions:', images.shape)
		print('Image label dimensions:', labels.shape)
		break
		
# =============================================================

n_images = 15
image_width = 32
num_channels = 3

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
												 sharex=True, sharey=True, figsize=(20, 2.5))
orig_images = features[:n_images]
decoded_images = decoded[:n_images]

for i in range(n_images):
		for ax, img in zip(axes, [orig_images, decoded_images]):
				curr_img = img[i].detach().to(torch.device('cpu'))
				#print(curr_img.shape)
				#ax[i].imshow(curr_img.view((image_width, image_width, num_channels)))
				ax[i].imshow(torch.transpose(curr_img.view((num_channels, image_width, image_width)), 0,2))

plt.savefig("train_decoded_images.png")

test_loader = DataLoader(dataset=test_dataset, 
												 batch_size=15, 
												 shuffle=True)

# Checking the dataset
for images, labels in test_loader:  
		print('Image batch dimensions:', images.shape)
		print('Image label dimensions:', labels.shape)
		break
		
# =============================================================

n_images = 15
image_width = 32
num_channels = 3

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
												 sharex=True, sharey=True, figsize=(20, 2.5))
orig_images = features[:n_images]
decoded_images = decoded[:n_images]

for i in range(n_images):
		for ax, img in zip(axes, [orig_images, decoded_images]):
				curr_img = img[i].detach().to(torch.device('cpu'))
				ax[i].imshow(torch.transpose(curr_img.view((num_channels, image_width, image_width)), 0,2))

plt.savefig("test_decoded_images.png")

"""## Scikit-learn Classifier

### On Original CIFAR-100
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier



train_loader = DataLoader(dataset=train_dataset, 
													batch_size=60000, 
													shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
													batch_size=10000, 
													shuffle=False)

for images, labels in train_loader:  
		print('Image batch dimensions:', images.shape)
		print('Image label dimensions:', labels.shape)
		break

num_channels = 3

image_dim = 32

X_train = np.array(images.reshape(50000, num_channels*image_dim*image_dim))
y_train = np.array(labels)


for images, labels in test_loader:  
		print('Image batch dimensions:', images.shape)
		print('Image label dimensions:', labels.shape)
		break

X_test = np.array(images.reshape(10000, num_channels*image_dim*image_dim))
y_test = np.array(labels)

rf = RandomForestClassifier(n_estimators=1, n_jobs=-1).fit(X_train, y_train)
print(f'Train Accuracy: {rf.score(X_train, y_train)*100}%')
print(f'Test Accuracy: {rf.score(X_test, y_test)*100}%')

"""### Using PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components=3*32)  # same size as autoencoder latent space
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf = RandomForestClassifier(n_estimators=1, n_jobs=-1).fit(X_train_pca, y_train)
print(f'Train Accuracy: {rf.score(X_train_pca, y_train)*100}%')
print(f'Test Accuracy: {rf.score(X_test_pca, y_test)*100}%')

"""### Auto-Encoder Compressed CIFAR-100"""

train_loader = DataLoader(dataset=train_dataset, 
													batch_size=1000, 
													shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
													batch_size=1000, 
													shuffle=False)

X_train_compr = np.ones((50000, num_hidden_1))
y_train = np.ones(50000)

start_idx = 0

for idx, (images, labels) in enumerate(train_loader): 
	features = images.view(-1, 3*32*32).to(device)
	decoded = model.encoder(features)
	X_train_compr[start_idx:start_idx+1000] = decoded.to(torch.device('cpu')).detach().numpy()
	y_train[start_idx:start_idx+1000] = labels
	start_idx += 1000

X_test_compr = np.ones((10000, num_hidden_1))
y_test = np.ones(10000)

start_idx = 0

for idx, (images, labels) in enumerate(test_loader): 
	features = images.view(-1, 3*32*32).to(device)
	decoded = model.encoder(features)
	X_test_compr[start_idx:start_idx+1000] = decoded.to(torch.device('cpu')).detach().numpy()
	y_test[start_idx:start_idx+1000] = labels
	start_idx += 1000

rf = RandomForestClassifier(n_estimators=1, n_jobs=-1).fit(X_train_compr, y_train)
print(f'Train Accuracy: {rf.score(X_train_compr, y_train)*100}%')
print(f'Test Accuracy: {rf.score(X_test_compr, y_test)*100}%')



fig= plt.figure(figsize=(15, 8))


# Plotted the accuracy Graph
def plot_accuracies(acc_1, acc_2, acc_3, labels, ests_list):

	plt.plot(ests_list, acc_1, '-rx')
	plt.plot(ests_list, acc_2, '-bx')
	plt.plot(ests_list, acc_3, '-gx')

	plt.legend(labels)

	plt.tight_layout()
	plt.xlabel('n_trees')
	plt.ylabel('accuracy %')
	plt.title('Accuracy vs. No. of Trees')

	plt.savefig("RF Accuracies v_s Num_Trees Part 3.png", bbox_inches='tight')




n_estimators = 1

rf_clf_Original = RandomForestClassifier(n_estimators= n_estimators, criterion='gini', max_depth= max(50, n_estimators/10), min_samples_split=30, n_jobs = -1, warm_start=True)
rf_clf_PCA = RandomForestClassifier(n_estimators= n_estimators, criterion='gini', max_depth= max(50, n_estimators/10), min_samples_split=30, n_jobs = -1, warm_start=True)
rf_clf_AE = RandomForestClassifier(n_estimators= n_estimators, criterion='gini', max_depth= max(50, n_estimators/10), min_samples_split=30, n_jobs = -1, warm_start=True)

orig_train_acc_scores = []
pca_train_acc_scores = []
ae_train_acc_scores = []


orig_test_acc_scores = []
pca_test_acc_scores = []
ae_test_acc_scores = []


ests_list = []

for n_estimators in [1, 10, 50, 100, 200, 500, 1000, 5000]:
#for n_estimators in [1, 10]:

	parameter_identifier_orig = f"RF_Original_P3_n_estimators={n_estimators}, max_depth= {rf_clf_Original.max_depth}.rfmodel"
	parameter_identifier_pca = f"RF_PCA_P3_n_estimators={n_estimators}, max_depth= {rf_clf_Original.max_depth}.rfmodel"
	parameter_identifier_ae = f"RF_AE_P3_n_estimators={n_estimators}, max_depth= {rf_clf_Original.max_depth}.rfmodel"
	
	rf_clf_Original.n_estimators = n_estimators
	rf_clf_PCA.n_estimators = n_estimators
	rf_clf_AE.n_estimators = n_estimators

	for filename, clf in [
					(parameter_identifier_orig, rf_clf_Original),
					(parameter_identifier_pca, rf_clf_PCA),
					(parameter_identifier_ae, rf_clf_AE)
						]:

		if os.path.exists(filename):
			with open(filename, "rb") as file:
				clf = pickle.load(file)
		else:
			if clf == rf_clf_Original:
				rf_clf_Original.fit(X_train, y_train)

			if clf == rf_clf_PCA:
				rf_clf_PCA.fit(X_train_pca, y_train)

			if clf == rf_clf_AE:
				rf_clf_AE.fit(X_train_compr, y_train)

			with open(filename, "wb") as file:
				pickle.dump(clf, file)


	orig_train_acc_scores.append(100.0 * rf_clf_Original.score(X_train, y_train))
	pca_train_acc_scores.append(100.0 * rf_clf_PCA.score(X_train_pca, y_train))
	ae_train_acc_scores.append(100.0 * rf_clf_AE.score(X_train_compr, y_train))


	orig_test_acc_scores.append(100.0 * rf_clf_Original.score(X_test, y_test))
	pca_test_acc_scores.append(100.0 * rf_clf_PCA.score(X_test_pca, y_test))
	ae_test_acc_scores.append(100.0 * rf_clf_AE.score(X_test_compr, y_test))


	ests_list.append(n_estimators)

	plt.subplot(1, 2, 1)
	plot_accuracies(orig_train_acc_scores, pca_train_acc_scores, ae_train_acc_scores,
		labels = ["Training - Original RF", "Training - PCA RF", "Training - AE RF"], ests_list = ests_list)
	plt.subplot(1, 2, 2)
	plot_accuracies(orig_test_acc_scores, pca_test_acc_scores, ae_test_acc_scores, 
		labels = ["Validation - Original RF", "Validation - PCA RF", "Validation - AE RF"], ests_list = ests_list)

	# {rf.score(X_train, y_train)*100}%
	# {rf.score(X_test, y_test)*100}%

	# {rf.score(X_train_pca, y_train)*100}%
	# {rf.score(X_test_pca, y_test)*100}%

	# {rf.score(X_train_compr, y_train)*100}%
	# {rf.score(X_test_compr, y_test)*100}%


