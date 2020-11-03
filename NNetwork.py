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


class NNetworkHelper:

	def __init__(self, train_folder, test_folder):
		self.train_patient_data_list = []
		self.test_patient_data_list = []
		self.batch_size = 5

		self._read_train_data(train_folder)
		self._read_test_data(test_folder)

		# More information about the model:
		# https://www.nature.com/articles/s41746-018-0065-x
		# The learning rate and decay is defined under 'Left Ventricular Hypertrophy Classification'

		self.model = CNN()
		self.criterion = nn.CrossEntropyLoss()  # The way we calculate the loss is defined here
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.025)

	def _read_train_data(self, data_folder):
		self._read_data(data_folder, self.train_patient_data_list)

	def _read_test_data(self, data_folder):
		self._read_data(data_folder, self.test_patient_data_list)

	def _read_data(self, data_folder, patient_data_list):
		patient_pickles = sorted(os.listdir(data_folder))
		for pickle_file in patient_pickles:
			pickle_file_path = os.path.join(data_folder, pickle_file)
			with open(pickle_file_path, "rb") as file:
				loaded_pickle = pickle.load(file)
				patient_data_list.append(loaded_pickle)

	def _convert_to_dataset(self, patient_data_list):
		# TODO: only ch2 systole is taken into account
		patient_dataset = [x.ch2_systole for x in patient_data_list]
		data_loader = DataLoader(patient_dataset, batch_size=self.batch_size)

	def preprocess_data(self):
		# lower resolution to 110*110
		# TODO: should probably also rotate here or smth
		for patient_data in itertools.chain(self.train_patient_data_list, self.test_patient_data_list):
			new_size = (110, 110)
			patient_data.ch2_systole = cv2.resize(patient_data.ch2_systole, new_size)
			patient_data.ch2_diastole = cv2.resize(patient_data.ch2_diastole, new_size)
			patient_data.ch4_systole = cv2.resize(patient_data.ch4_systole, new_size)
			patient_data.ch4_diastole = cv2.resize(patient_data.ch4_diastole, new_size)
			patient_data.lvot_systole = cv2.resize(patient_data.lvot_systole, new_size)
			patient_data.lvot_diastole = cv2.resize(patient_data.lvot_diastole, new_size)

	def train(self, num_epochs):
		loss_list = []
		for epoch in range(num_epochs):
			for patient_data in self.train_patient_data_list:
				# TODO: currently only training on the ch2 systole
				output = self.model(patient_data.ch2_systole)
				loss = self.criterion(output, patient_data.hypertrophic)
				loss_list.append(loss.item())

				# Backprop and Adam optimisation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

	# TODO: idk hogy ez finished e

	def test(self):
		self.model.eval()
		with torch.no_grad():
			# TODO: other statistics, like average difference, max difference etc.
			correct = 0
			total = 0
			for patient_data in self.test_patient_data_list:
				# TODO: currently only testing on the ch2 systole
				output = self.model(patient_data.ch2_systole)
				result_diff = output.data - int(patient_data.hypertrophic)



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
			print("tf what even is transforming")
			sample = self.transform(sample)

		return sample
