import itertools
import operator
import os
import pickle

import numpy as np
import pydicom as dicom

import collections


class VectorAverageCalc:
	def __init__(self):
		self.pickledDataExists = True
		normal_list = []
		if not self.pickledDataExists:
			self.hypertrophy_folder = "../hypertrophy"
			hyp_folders = sorted(os.listdir(self.hypertrophy_folder))

			for patient_folder in hyp_folders:
				folder_path = os.path.join(self.hypertrophy_folder, patient_folder, "la")
				dcm_files = sorted(os.listdir(folder_path))

				dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
				if len(dcm_files) == 0:  # sometimes the order number is missing at the end
					dcm_files = sorted(os.listdir(folder_path))

				# we only care about files that have a filename longer than 10 characters
				dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) > 10]

				vector_list_in_folder = []

				# print(patient_folder)

				for file in dcm_files:
					try:
						temp_ds = dicom.dcmread(os.path.join(folder_path, file))
						# We only want to store each orientation once
						if temp_ds.ImageOrientationPatient not in vector_list_in_folder:
							vector_list_in_folder.append(temp_ds.ImageOrientationPatient)
					except Exception as e:
						print('Exception while reading dcm file' + os.path.join(folder_path, file) + ': ' + repr(e))

				# We'll only save the ones that have three different normals in it, which are probably the ones that interest us the most
				if len(vector_list_in_folder) == 3:
					# This can probably be done in a much cleaner way
					reshaped_vector_list_in_folder = np.asarray(vector_list_in_folder).reshape(3, 2, 3)
					normal_list_in_folder = np.empty([3, 3])
					for i in range(3):
						normal_list_in_folder[i] = np.cross(reshaped_vector_list_in_folder[i][0], reshaped_vector_list_in_folder[i][1])
					normal_list.append(normal_list_in_folder)
					# print(patient_folder)

			file = open("pickled_data_limited", 'wb')
			pickle.dump(normal_list, file)
			file.close()
		else: # pickled data exists
			file = open("pickled_data_limited", 'rb')
			normal_list = pickle.load(file)
			file.close()
		# print(normal_list)
		for i in range(3):
			for normal in normal_list:
				# if len(normal) > 1:
				print(normal[1][i])
			print("\n")
