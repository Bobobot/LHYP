import os
import pickle
from enum import Enum

import pydicom as dicom
import numpy as np

from dicom_data_holder import DicomDataHolder


class ChamberEnum(Enum):
	NOTHING, CH2, CH4, LVOT = range(4)


class ChamberVectorEnum(Enum):
	NOTHING, CH2_VEC, CH4_VEC, LVOT_VEC, CH2_VEC_INVERTED, CH4_VEC_INVERTED, LVOT_VEC_INVERTED = range(7)


class HeartPhaseEnum(Enum):
	NOTHING, SYSTOLE, DIASTOLE = range(3)


class DicomPreprocessor:

	def __init__(self, root_folder):
		self.root_folder = root_folder
		# The amount that a normal can differ from the average normal (seen below) to still be considered
		self.NORMAL_DIFFERENCE_THRESHOLD = 0.4
		# The file which the preprocessor's progress gets saved to
		self.progress_file = "progress.rick"

		# We'll create the data folder that the patients' data will reside in
		if not os.path.exists('data'):
			os.makedirs('data')

		# Constant vectors for chamber recognition
		# These are the normals of the plane for each chamber, which is an average calculated from the example database
		ch2_vec = np.array([0.6636733448, 0.7304146021, 0.0420980024])
		ch4_vec = np.array([0.0415284776, -0.6452731427, 0.7458425355])
		lvot_vec = np.array([-0.6734138762, -0.1534521711, -0.7004136136])
		ch2_vec_negative = np.negative(ch2_vec)
		ch4_vec_negative = np.negative(ch4_vec)
		lvot_vec_negative = np.negative(lvot_vec)

		self._chamber_vectors = {
			ChamberVectorEnum.CH2_VEC: ch2_vec,
			ChamberVectorEnum.CH4_VEC: ch4_vec,
			ChamberVectorEnum.LVOT_VEC: lvot_vec,
			ChamberVectorEnum.CH2_VEC_INVERTED: ch2_vec_negative,
			ChamberVectorEnum.CH4_VEC_INVERTED: ch4_vec_negative,
			ChamberVectorEnum.LVOT_VEC_INVERTED: lvot_vec_negative
		}

		self._reset_state()

	# This method is responsible for resetting crucial things between different patients
	def _reset_state(self):
		# The chamber counter counts how many pictures we've seen of each chamber. We need this so we can identify
		# both the systole and diastole.
		self._chamber_counter = {
			ChamberEnum.CH2: 0,
			ChamberEnum.CH4: 0,
			ChamberEnum.LVOT: 0
		}

		self.data_holder = DicomDataHolder()

	def process(self):
		# Go through all the folders in the root folder
		hyp_folders = sorted(os.listdir(self.root_folder))

		# Try to load the list of already processed patients if exist
		hyp_folders = self._load_state(hyp_folders)

		# The patient folder is the one with the numbers (eg 10129327AMR806)
		for patient_folder in hyp_folders:
			# We read in whether this current dataset is of a hypertrophic patient or not, and set the data holder's value accordingly.
			# If the patient is an athlete, or is 18 years of age, we skip their dataset
			skip, hypertrophic = self._is_hypertrophic(patient_folder)
			if skip: continue
			self.data_holder.hypertrophic = hypertrophic
			folder_path = os.path.join(self.root_folder, patient_folder, "la")

			# This is where we do the actual processing - the _get_classification method classifies the image, which then we
			# TODO: finish comment
			dcm_files = sorted(os.listdir(folder_path))
			for file_name in dcm_files:
				dcm_file = dicom.dcmread(os.path.join(folder_path, file_name))
				chamber_view_type, negative, heart_phase = self._get_classification(dcm_file)
				# debug shit, remove
				# if chamber_view_type == ChamberEnum.CH2:
					# print(f'{patient_folder}\\{file_name} | inverted: {negative}')
					# break
				# end of debug shit
				pixel_data = self._process_image(dcm_file, chamber_view_type, negative)
				self._update_data_holder(pixel_data, chamber_view_type, heart_phase)

			# Save the list of already processed patients to a file, and dumps the current patient's data holder
			self._save_state(patient_folder)

			# After we're done processing this patient, we reset the state in order to prepare for the next one
			self._reset_state()

		print("Processing finished.")

	# returns a pair of what view the scan is (how many chambers there are), and whether the specific image is of a systole, a diastole, or nothing
	def _get_classification(self, dcm_file):
		instance_number = dcm_file.InstanceNumber
		# Reshape is necessary because the image orientation is stored as a 6 long vector, which we will need to turn into a single normal
		image_orientation_patient = np.asarray(dcm_file.ImageOrientationPatient).reshape(2, 3)

		# eww
		negative = False
		chamber_view_type: ChamberEnum
		closest_vector = self._find_closest_vector(image_orientation_patient)

		if closest_vector == ChamberVectorEnum.NOTHING: return ChamberEnum.NOTHING, False, HeartPhaseEnum.NOTHING

		if closest_vector == ChamberVectorEnum.CH2_VEC:
			chamber_view_type = ChamberEnum.CH2
		elif closest_vector == ChamberVectorEnum.CH2_VEC_INVERTED:
			chamber_view_type = ChamberEnum.CH2
			negative = True
		elif closest_vector == ChamberVectorEnum.CH4_VEC:
			chamber_view_type = ChamberEnum.CH4
		elif closest_vector == ChamberVectorEnum.CH4_VEC_INVERTED:
			chamber_view_type = ChamberEnum.CH4
			negative = True
		elif closest_vector == ChamberVectorEnum.LVOT_VEC:
			chamber_view_type = ChamberEnum.LVOT
		elif closest_vector == ChamberVectorEnum.LVOT_VEC_INVERTED:
			chamber_view_type = ChamberEnum.LVOT
			negative = True
		else:
			print("Incorrect closest vector value")

		# We make the (reasonable) assumption that if a one of the patient's scan of a specific chamber is inverted,
		# then the rest of their scans of the same chamber will be inverted too
		self._chamber_counter[chamber_view_type] += 1
		heart_phase = HeartPhaseEnum.NOTHING
		if self._chamber_counter[chamber_view_type] == 9:
			heart_phase = HeartPhaseEnum.SYSTOLE
		elif self._chamber_counter[chamber_view_type] == 24:
			heart_phase = HeartPhaseEnum.DIASTOLE
		return chamber_view_type, negative, heart_phase

	# clips the top and bottom percentile, converts the datatype to uint8, and mirrors & rotates the image if needed
	def _process_image(self, dcm_file, chamber_view_type, negative):
		pixel_array = dcm_file.pixel_array
		# We clip the bottom and upper 10% of the pixel data
		processed_pixel_array = np.clip(pixel_array, np.percentile(pixel_array, 10), np.percentile(pixel_array, 90))
		# We convert the pixel array to store uint8 values instead of uint16, therefore saving space
		processed_pixel_array *= (255.0 / processed_pixel_array.max())
		processed_pixel_array = processed_pixel_array.astype(np.uint8)
		if negative:
			if chamber_view_type == ChamberEnum.CH2:
				pass  # TODO mirror and rotate image
			elif chamber_view_type == ChamberEnum.CH4:
				pass  # TODO mirror and rotate image
			elif chamber_view_type == ChamberEnum.LVOT:
				pass  # TODO mirror and rotate image
		return processed_pixel_array

	def _find_closest_vector(self, image_orientation_patient):
		image_orientation_normal = np.cross(image_orientation_patient[0], image_orientation_patient[1])
		vec_distance_values = map(lambda vec: np.linalg.norm(image_orientation_normal - vec), self._chamber_vectors.values())
		# This dictionary stores the image orientation normal's distance to every predefined chamber normal
		vec_distances = dict(zip(self._chamber_vectors.keys(), vec_distance_values))
		# We don't want to mistake this vector for a chamber view that we aren't even looking for
		if min(vec_distances.values()) <= self.NORMAL_DIFFERENCE_THRESHOLD:
			# We find which chamber view's normal this vector is closest to and return its name
			return min(vec_distances, key=vec_distances.get)
		return ChamberVectorEnum.NOTHING

	def _is_hypertrophic(self, patient_folder):
		meta_file = open(os.path.join(self.root_folder, patient_folder, "meta.txt"), 'rb')
		file_content = meta_file.read()
		meta_file.close()
		skip = file_content in ("adult_m_sport", "adult_f_sport", "U18_f", "U18_m")  # if any of these are true then we can't check hypertrophy
		hypertrophic = file_content != "Normal"
		return skip, hypertrophic

	def _update_data_holder(self, pixel_data, chamber_view_type, heart_phase):
		if heart_phase == HeartPhaseEnum.NOTHING: return  # If it's not a systole or diastole, we don't want to save it
		systole = heart_phase == HeartPhaseEnum.SYSTOLE  # Just to shorten the code a little

		# Ugly, but it do be like that sometimes
		if chamber_view_type == ChamberEnum.CH2:
			if systole:
				self.data_holder.ch2_systole = pixel_data
			else:
				self.data_holder.ch2_diastole = pixel_data
		elif chamber_view_type == ChamberEnum.CH4:
			if systole:
				self.data_holder.ch4_systole = pixel_data
			else:
				self.data_holder.ch4_diastole = pixel_data
		elif chamber_view_type == ChamberEnum.LVOT:
			if systole:
				self.data_holder.lvot_systole = pixel_data
			else:
				self.data_holder.lvot_diastole = pixel_data

	def _save_state(self, patient_folder):
		with open(f'data/{patient_folder}.rick', "wb") as file:
			pickle.dump(self.data_holder, file)

		with open(self.progress_file, "a") as file:
			file.write(f'{patient_folder}\n')

	def _load_state(self, hyp_folders):
		if os.path.isfile(self.progress_file):
			with open(self.progress_file, "r") as file:
				for line in file:
					line = line.strip()  # preprocess line
					# We remove all patients from the folder list whose data has already been processed
					hyp_folders.remove(line)
		return hyp_folders
