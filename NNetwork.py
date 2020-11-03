import glob
import itertools
import os
import pickle
import cv2
import scipy
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class NNetworkHelper:

	def __init__(self, train_folder, test_folder):
		self.batch_size = 5
		# The transformation to be performed on every input image
		# We normalize, and turn our numpy array (0-255) to a tensor (0.0-1.0)
		trans = transforms.Compose([
			# We lower the resolution to 110*110, according to the paper
			transforms.Resize((110, 110)),
			# These numbers were roughly approximated from a randomly chosen sample
			transforms.Normalize(mean=40, std=60),
			transforms.ToTensor()
		])
		train_dataset = HeartDataSet(train_folder, trans)
		test_dataset = HeartDataSet(test_folder, trans)

		self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

		# More information about the model:
		# https://www.nature.com/articles/s41746-018-0065-x
		# The learning rate and decay is defined under 'Left Ventricular Hypertrophy Classification'

		self.model = CNN()
		self.criterion = nn.CrossEntropyLoss()  # The way we calculate the loss is defined here
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.025)

	def train(self, num_epochs):
		loss_list = []
		for epoch in range(num_epochs):
			for i, (images, are_hypertrophic) in enumerate(self.train_loader):
				outputs = self.model(images)
				loss = self.criterion(outputs, are_hypertrophic)
				loss_list.append(loss.item())

				# Backprop and Adam optimisation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# TODO: debug wtf this is
				print(outputs.data)
				# Track the accuracy
				# total = labels.size(0)
				# _, predicted = torch.max(outputs.data, 1)
				# correct = (predicted == labels).sum().item()
				# acc_list.append(correct / total)

				# TODO: print smth here

	# TODO: idk hogy ez finished e

	def test(self):
		self.model.eval()
		with torch.no_grad():
			# TODO: other statistics, like average difference, max difference etc.
			correct = 0
			total = 0
			for images, are_hypertrophic in self.test_loader:
				outputs = self.model(images)
				# TODO: calculate results and stuff, currently idk wtf outputs.data even is
				# result_diff = output.data - int(patient_data.hypertrophic)

				# TODO: print smth here

		# TODO: save the model


class CNN(nn.Module):

	# CNN model is from:
	# https://www.nature.com/articles/s41746-018-0065-x/figures/13
	def __init__(self):
		super(CNN, self).__init__()

		# TODO: find out kernel size used in paper
		kersiz = 3
		pad = kersiz - 1

		# 64x layer

		self.layer1 = nn.Conv2d(1, 64, kernel_size=kersiz, stride=1, padding=pad)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(64))
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=kersiz, stride=2, padding=pad),
			nn.BatchNorm2d(128),
			nn.Dropout2d(0.4)
		)

		# 128x layer

		self.layer4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(128)
		)
		self.layer5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(128)
		)
		self.layer6 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=kersiz, stride=2, padding=pad),
			nn.BatchNorm2d(256),
			nn.Dropout2d(0.4)
		)

		# 256x layer

		self.layer7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(256)
		)
		self.layer8 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(256)
		)
		self.layer9 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=kersiz, stride=2, padding=pad),
			nn.BatchNorm2d(512),
			nn.Dropout2d(0.4)
		)

		# 512x layer

		self.layer10 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(512)
		)
		self.layer11 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(512)
		)
		self.layer12 = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=kersiz, stride=2, padding=pad),
			nn.BatchNorm2d(1024),
			nn.Dropout2d(0.4)
		)

		# 1024x layer

		self.layer13 = nn.Sequential(
			nn.Conv2d(1024, 1024, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(1024)
		)
		self.layer14 = nn.Sequential(
			nn.Conv2d(1024, 1024, kernel_size=kersiz, stride=1, padding=pad),
			nn.BatchNorm2d(1024)
		)
		self.layer15 = nn.Sequential(
			nn.Conv2d(1024, 1024, kernel_size=kersiz, stride=2, padding=pad),
			nn.BatchNorm2d(1024),
			nn.Dropout2d(0.4)
		)
		self.layer16 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(0.4)
		)

		# one extra layer that is not in the original model
		self.layer17 = nn.Linear(3 * 3 * 1024, 500)

		self.layer18 = nn.Sequential(
			nn.Linear(500, 1),
			nn.Sigmoid()
		)

	def forward(self, x: nn.Tensor):
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
		out = self.layer17(out)
		out = self.layer18(out)
		return out


class HeartDataSet(Dataset):
	def __init__(self, data_folder, transform=None):
		self.data_folder = data_folder
		self.transform = transform

		def __len__(self):
			# TODO: test this because god knows at this point
			return len(glob.glob1(self.data_folder, "*.rick"))

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		pickle_file = sorted(os.listdir(self.data_folder))[index]
		loaded_pickle = 0
		with open(os.path.join(self.data_folder, pickle_file), "rb") as file:
			loaded_pickle = pickle.load(file)

		# TODO: only taking ch2 systole into account
		sample = {'image': loaded_pickle.ch2_systole, 'hypertrophic': loaded_pickle.hypertrophic}

		if self.transform:
			sample = self.transform(sample)

		return sample
