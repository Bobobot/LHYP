import glob
import itertools
import os
import pickle
import cv2
import numpy as np
import sklearn
import sklearn.metrics
import torch
import matplotlib.pyplot as plt

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
		self.learning_rate = 0.00001
		# self.lr_decay = 0.85

		# The transformation to be performed on every input image
		# We crop, resize, and then turn our numpy array (0-255) to a tensor (0.0-1.0)
		trans = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((240, 240)),
			transforms.CenterCrop((150, 150)),
			# We lower the resolution to 110*110, according to the paper
			transforms.Resize((110, 110)),
			transforms.ToTensor(),
			# These numbers were roughly approximated from a randomly chosen sample
		])

		# TODO: seperate train, validation and test folders manually
		full_dataset = nc.SafeDataset(HeartDataSet(data_folder, trans))
		train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [675, 135])

		# train_dataset, test_dataset = sklearn.model_selection.train_test_split(full_dataset, 0.25, 0.75, random_state=1, stratify=)

		# train_dataset = nc.SafeDataset(HeartDataSet(train_folder, trans))
		# test_dataset = nc.SafeDataset(HeartDataSet(test_folder, trans))

		self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

		# More information about the model:
		# https://www.nature.com/articles/s41746-018-0065-x
		# The learning rate and decay is defined under 'Left Ventricular Hypertrophy Classification'

		self.model = CNN().to(self.device)
		self.criterion = nn.BCELoss()  # The way we calculate the loss is defined here
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,)
		# self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

		# variables defined here so we preserve data between validation
		self.validate_loss_list = []
		self.validate_acc_list = []
		self.validate_f1score_list = []

	def train(self, num_epochs):
		print('Starting training...')
		total_step = len(self.train_loader)
		loss_list = []
		acc_list = []
		f1score_list = []
		# data for f1 score
		true_pos = 0
		false_pos = 0
		false_neg = 0
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

				# calc classification report
				# print(sklearn.metrics.classification_report(target_array, result_array.round(), target_names=['False', 'True']))

				# Track accuracy and data for f1-score
				total = target_array.size
				for batch_num in range(total):
					difference = abs(target_array[batch_num] - result_array[batch_num])
					accuracy = 1 - difference
					batch_acc_list.append(accuracy)

					# f1-score
					if target_array[batch_num] == 1:
						# If the accuracy is larger than 0.5, the model has guessed successfully
						if accuracy > 0.5:
							true_pos += 1
						else:
							false_neg += 1
					else:  # target is false
						if accuracy < 0.5:
							false_pos += 1  # we guessed true even though we shouldn't have

			# Calculate F1-score
			precision = true_pos / (true_pos + false_pos)
			recall = true_pos / (true_pos + false_neg)
			f1_score = 2 * (precision * recall) / (precision + recall)
			f1score_list.append(f1_score)

			print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.average(batch_loss_list):.4f}, Accuracy: {np.average(batch_acc_list) * 100:.2f}%, F1-score: {f1_score:.3f}')
			loss_list.append(np.average(batch_loss_list))
			acc_list.append(np.average(batch_acc_list))

			self.validate()

			self._draw_plots(loss_list, f1score_list)
			# self.scheduler.step()  # apply learning rate decay

	def validate(self):
		self.model.eval()

		# data for f1 score
		true_pos = 0
		false_pos = 0
		false_neg = 0
		with torch.no_grad():
			batch_loss_list = []
			batch_acc_list = []
			for batch in self.test_loader:
				images = batch["image"].to(self.device)
				are_hypertrophic = batch["hypertrophic"].to(self.device)
				outputs = self.model(images)
				loss = self.criterion(outputs, are_hypertrophic.float())
				batch_loss_list.append(loss.item())

				result_array = outputs.cpu().data.numpy()
				target_array = are_hypertrophic.cpu().data.numpy().astype(int)

				# Track accuracy and data for f1-score
				total = target_array.size
				for batch_num in range(total):
					difference = abs(target_array[batch_num] - result_array[batch_num])
					accuracy = 1 - difference
					batch_acc_list.append(accuracy)

					# f1-score
					if target_array[batch_num] == 1:
						# If the accuracy is larger than 0.5, the model has guessed successfully
						if accuracy > 0.5:
							true_pos += 1
						else:
							false_neg += 1
					else:  # target is false
						if accuracy < 0.5:
							false_pos += 1  # we guessed true even though we shouldn't have

			# Calculate F1-score
			precision = true_pos / (true_pos + false_pos)
			recall = true_pos / (true_pos + false_neg)
			f1_score = 2 * (precision * recall) / (precision + recall)
			self.validate_f1score_list.append(f1_score)

			self.validate_loss_list.append(np.average(batch_loss_list))
			self.validate_acc_list.append(np.average(batch_acc_list))

			print(f'Testing F1-score: {f1_score:.3f}')

		torch.save(self.model.state_dict(), "cnn_model.torch")

	def _draw_plots(self, loss_list, f1score_list):
		l_fig, l_ax = plt.subplots()
		l_ax.plot(loss_list, label='Loss (train)', color='red')
		l_ax.plot(self.validate_loss_list, label='Loss (validation)', color='purple')
		l_ax.set_xlabel('Epoch')
		l_ax.set_ylabel('Loss')
		handles, labels = l_ax.get_legend_handles_labels()
		l_ax.legend(handles, labels)

		f_fig, f_ax = plt.subplots()
		f_ax.plot(f1score_list, label='F1 score (train)', color='green')
		f_ax.plot(self.validate_f1score_list, label='F1 score (validation)', color='blue')
		f_ax.set_xlabel('Epoch')
		f_ax.set_ylabel('F1 score')
		handles, labels = f_ax.get_legend_handles_labels()
		f_ax.legend(handles, labels)

		l_fig.show()
		f_fig.show()
		l_fig.savefig("plots/loss_plot.png")
		f_fig.savefig("plots/f1_plot.png")


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
		self._num_of_patient_files = len(glob.glob1(self.data_folder, "*.rick"))
		# How many images a patient's file has
		# For example, if you want to use both the ch2 systole and ch2 diastole of a patient,
		# this value would be 2
		self.patient_images = 2

	def __len__(self):
		return self._num_of_patient_files * self.patient_images

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		# iteration refers to which type of image we're currently indexing. (chamber view + systole/diastole)
		# the true index is the index inside the collection of those specific images.
		iteration = np.math.floor(index / self._num_of_patient_files)
		true_index = index % self._num_of_patient_files

		pickle_file = sorted(os.listdir(self.data_folder))[true_index]
		loaded_pickle = 0
		with open(os.path.join(self.data_folder, pickle_file), "rb") as file:
			loaded_pickle = pickle.load(file)

		# TODO: only taking ch2 into account (maybe this'll remain idk)
		if iteration == 0:
			sample = {'image': loaded_pickle.ch2_systole['pixel_data'], 'hypertrophic': loaded_pickle.hypertrophic}
		elif iteration == 1:
			sample = {'image': loaded_pickle.ch2_diastole['pixel_data'], 'hypertrophic': loaded_pickle.hypertrophic}
		else:
			print(f'Error while trying to read .rick file: iteration count too big')
			print(f'index: {index}')
			print(f'true_index: {true_index}')
			print(f'iteration: {iteration}')
			print(f'num of patient images: {self._num_of_patient_files}')
		# print(loaded_pickle.hypertrophic)

		# display image before transformation
		# cv2.imshow('before', sample['image'])

		if self.transform:
			sample['image'] = self.transform(sample['image'])

		# display image after transformation
		# cv2.imshow('after', sample['image'].detach().numpy()[0])
		# cv2.waitKey()

		return sample
