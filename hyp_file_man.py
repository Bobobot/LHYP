import math
import os
import pickle
import random
import shutil


class HypertrophyFileManager:
	def separate_files(self, data_folder, percentages, folder_names, max_acceptable_ratio_difference):
		pickle_files = sorted(os.listdir(data_folder))
		pickle_files = [file for file in pickle_files if file.endswith(".rick")]

		if len(percentages) != len(folder_names):
			print("ERROR: percentage and folder name list must be the same length")
			return
		if sum(percentages) != 100:
			print("ERROR: sum of percentages must be 100")
			return

		file_numbers = []
		for perc in percentages:
			file_numbers.append(math.floor(len(pickle_files) * (perc / 100)))

		for folder_name in folder_names:
			folder_path = os.path.join(data_folder, folder_name)
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)

		ratios_good = False
		attempt_count = 0
		list_of_file_lists = []
		while not ratios_good:
			pickle_files_copy = pickle_files.copy()
			list_of_file_lists.clear()
			ratios_good = True
			print(f'attempt {attempt_count + 1}')
			attempt_count += 1

			list_of_ratios = []
			for file_number in file_numbers:
				list_of_ratios.clear()
				list_of_files = random.sample(pickle_files_copy, file_number)
				pickle_files_copy = [file for file in pickle_files_copy if file not in list_of_files]
				list_of_file_lists.append(list_of_files)
			for file_list in list_of_file_lists:
				_, _, ratio = self.check_ratio_list(data_folder, file_list)
				list_of_ratios.append(ratio)
				if not (0.5 - max_acceptable_ratio_difference < ratio < 0.5 + max_acceptable_ratio_difference):
					ratios_good = False
					break
			if ratios_good:
				print("Final ratios:")
				for i, folder_name in enumerate(folder_names):
					print(f'{folder_name}: {list_of_ratios[i]}')

		for folder_index, file_list in enumerate(list_of_file_lists):
			for file_name in file_list:
				file_path = os.path.join(data_folder, file_name)
				if os.path.isfile(file_path):
					shutil.move(file_path, os.path.join(data_folder, folder_names[folder_index]))


	def check_ratio_list(self, data_folder, file_list):
		hypertrophic_count = 0
		total = 0

		for pickle_file in file_list:
			with open(os.path.join(data_folder, pickle_file), "rb") as file:
				loaded_pickle = pickle.load(file)
				total += 1
				if loaded_pickle.hypertrophic:
					hypertrophic_count += 1

		return total, hypertrophic_count, hypertrophic_count / total

	def check_ratio_folder(self, data_folder):
		pickle_files = sorted(os.listdir(data_folder))
		total, hypertrophic_count, ratio = self.check_ratio_list(data_folder, pickle_files)
		print(f'Total files: {total}\nHypertrophic files: {hypertrophic_count}\nRatio: {hypertrophic_count / total}')
