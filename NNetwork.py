import glob
import itertools
import os
import pickle
import cv2
import numpy as np
import sklearn
import torch

import torch.nn as nn
import torch.nn.functional as F
import nonechucks as nc
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class NNetworkHelper:

	def __init__(self, data_folder):
		# initialize CUDA
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(f'Pytorch will utilize the following device: {self.device}')

		self.batch_size = 5

		# TODO: normalization creates barely visible pictures, fix it
		# The transformation to be performed on every input image
		# We normalize, and turn our numpy array (0-255) to a tensor (0.0-1.0)
		trans = transforms.Compose([
			transforms.ToPILImage(),
			# We lower the resolution to 110*110, according to the paper
			transforms.Resize((110, 110)),
			transforms.ToTensor(),
			# These numbers were roughly approximated from a randomly chosen sample
			# transforms.Normalize(mean=40, std=60)
		])
		full_dataset = nc.SafeDataset(HeartDataSet(data_folder, trans))
		train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [375, 75])

		# train_dataset, test_dataset = sklearn.model_selection.train_test_split(full_dataset, 0.25, 0.75, random_state=1, stratify=)

		# train_dataset = nc.SafeDataset(HeartDataSet(train_folder, trans))
		# test_dataset = nc.SafeDataset(HeartDataSet(test_folder, trans))

		self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)  # TODO: shuffle true
		self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

		# More information about the model:
		# https://www.nature.com/articles/s41746-018-0065-x
		# The learning rate and decay is defined under 'Left Ventricular Hypertrophy Classification'

		self.model = CNN().to(self.device)
		self.criterion = nn.BCELoss()  # The way we calculate the loss is defined here
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.025)

	def train(self, num_epochs):
		print('Starting training...')
		total_step = len(self.train_loader)
		loss_list = []
		acc_list = []
		f1score_list = []
		for epoch in range(num_epochs):
			batch_loss_list = []
			batch_acc_list = []
			for i, batch, in enumerate(self.train_loader):
				images = batch["image"].to(self.device)
				are_hypertrophic = batch["hypertrophic"].to(self.device)

				# get network outputs and calc loss
				outputs = self.model(images)
				loss = self.criterion(outputs, are_hypertrophic.float())
				batch_loss_list.append(loss.item())

				# backprop and Adam optimisation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# converting to numpy arrays
				result_array = outputs.cpu().data.numpy()
				target_array = are_hypertrophic.cpu().data.numpy().astype(int)

				# Track the accuracy
				total = target_array.size
				difference = 0
				for batch_num in range(total):
					difference += abs(target_array[batch_num] - result_array[batch_num])
				accuracy = 1 - (difference / total)
				batch_acc_list.append(accuracy)

			print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.average(batch_loss_list):.4f}, Accuracy: {np.average(batch_acc_list) * 100:.2f}%')
			loss_list.extend(batch_loss_list)
			acc_list.extend(batch_acc_list)

	# TODO: idk hogy ez finished e

	def test(self):
		print('\nStarting testing...')
		acc_list = []
		self.model.eval()
		with torch.no_grad():
			# TODO: other statistics, like average difference, max difference etc.
			correct = 0
			total = 0
			for batch in self.test_loader:
				images = batch["image"].to(self.device)
				are_hypertrophic = batch["hypertrophic"].to(self.device)
				outputs = self.model(images)

				result_array = outputs.cpu().data.numpy()
				target_array = are_hypertrophic.cpu().data.numpy().astype(int)
				total = target_array.size
				difference = 0
				for batch_num in range(total):
					difference += abs(target_array[batch_num] - result_array[batch_num])
				accuracy = 1 - (difference / total)
				acc_list.append(accuracy)

			print(f'Final model accuracy: {np.average(acc_list) * 100:.2f}%')

		torch.save(self.model.state_dict(), "cnn_model.torch")


class CNN(nn.Module):

	# CNN model is from:
	# https://www.nature.com/articles/s41746-018-0065-x/figures/13
	def __init__(self):
		super(CNN, self).__init__()

		# Kernel size is specified as 3 in the paper
		kersiz = 3

		# 64x layer

		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(64))
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=kersiz, stride=2, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(128),
			nn.Dropout2d(0.4)
		)

		# 128x layer

		self.layer4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(128)
		)
		self.layer5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(128)
		)
		self.layer6 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=kersiz, stride=2, padding=0),
			nn.LeakyReLU(),
			nn.BatchNorm2d(256),
			nn.Dropout2d(0.4)
		)

		# 256x layer

		self.layer7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(256)
		)
		self.layer8 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(256)
		)
		self.layer9 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=kersiz, stride=2, padding=0),
			nn.LeakyReLU(),
			nn.BatchNorm2d(512),
			nn.Dropout2d(0.4)
		)

		# 512x layer

		self.layer10 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(512)
		)
		self.layer11 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(512)
		)
		self.layer12 = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=kersiz, stride=2, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(1024),
			nn.Dropout2d(0.4)
		)

		# 1024x layer

		self.layer13 = nn.Sequential(
			nn.Conv2d(1024, 1024, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(1024)
		)
		self.layer14 = nn.Sequential(
			nn.Conv2d(1024, 1024, kernel_size=kersiz, stride=1, padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(1024)
		)
		self.layer15 = nn.Sequential(
			nn.Conv2d(1024, 1024, kernel_size=kersiz, stride=1, padding=1),
			nn.BatchNorm2d(1024),
			nn.Dropout2d(0.4)
		)
		self.layer16 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Dropout2d(0.4)
		)

		# one extra layer that is not in the original model
		self.layer17 = nn.Linear(3 * 3 * 1024, 500)

		self.layer18 = nn.Sequential(
			nn.Linear(500, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		out = self.layer8(out)
		out = self.layer9(out)
		out = self.layer10(out)
		out = self.layer11(out)
		out = self.layer12(out)
		out = self.layer13(out)
		out = self.layer14(out)
		out = self.layer15(out)
		out = self.layer16(out)
		out = self.layer17(out.view(out.size(0), -1))
		out = self.layer18(out)
		return torch.squeeze(out)


class HeartDataSet(Dataset):
	def __init__(self, data_folder, transform=None):
		self.data_folder = data_folder
		self.transform = transform

	def __len__(self):
		return len(glob.glob1(self.data_folder, "*.rick"))

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		pickle_file = sorted(os.listdir(self.data_folder))[index]
		loaded_pickle = 0
		with open(os.path.join(self.data_folder, pickle_file), "rb") as file:
			loaded_pickle = pickle.load(file)

		# TODO: only taking ch2 systole into account
		sample = {'image': loaded_pickle.ch2_systole['pixel_data'], 'hypertrophic': loaded_pickle.hypertrophic}
		# print(loaded_pickle.hypertrophic)

		if self.transform:
			sample['image'] = self.transform(sample['image'])

		return sample