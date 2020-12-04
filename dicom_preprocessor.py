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

# EXTRA refers to every xth picture that we save between the systole and diastole so that we have more data to work with
class HeartPhaseEnum(Enum):
	NOTHING, SYSTOLE, DIASTOLE, EXTRA = range(4)

# Usage: Call the process() method of this class to start processing the folder that was given in the constructor
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
			print(f'Processing patient {patient_folder}...')
			# We read in whether this current dataset is of a hypertrophic patient or not, and set the data holder's value accordingly.
			# If the patient is an athlete, or is 18 years of age, we skip their dataset
			skip, hypertrophic = self._is_hypertrophic(patient_folder)
			if skip: continue
			self.data_holder.hypertrophic = hypertrophic
			folder_path = os.path.join(self.root_folder, patient_folder, "la")

			# This is where we do the actual processing - the _get_classification method classifies the image, which then
			# we process, and dump into a pickle file.
			dcm_files = sorted(os.listdir(folder_path))
			failed = False
			for file_name in dcm_files:
				try:
					dcm_file = dicom.dcmread(os.path.join(folder_path, file_name))
					chamber_view_type, negative, heart_phase = self._get_classification(dcm_file)
					pixel_data = self._process_image(dcm_file, chamber_view_type, negative)
				except BaseException as e:
					failed = True
					print(f'Error while trying to process file {file_name}.')
					break
				self._update_data_holder(dcm_file, pixel_data, negative, chamber_view_type, heart_phase)

			# Save the list of already processed patients to a file, and dumps the current patient's data holder
			if not failed: self._save_state(patient_folder)

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
		cham_count = self._chamber_counter[chamber_view_type]
		heart_phase = HeartPhaseEnum.NOTHING
		if cham_count == 9:
			heart_phase = HeartPhaseEnum.SYSTOLE
		elif cham_count == 24:
			heart_phase = HeartPhaseEnum.DIASTOLE
		elif 9 < cham_count < 24 and cham_count % 3 == 0:
			heart_phase = HeartPhaseEnum.EXTRA
		return chamber_view_type, negative, heart_phase

	# clips the top and bottom percentile, converts the datatype to uint8
	def _process_image(self, dcm_file, chamber_view_type, negative):
		pixel_array = dcm_file.pixel_array
		# We clip the bottom and upper 10% of the pixel data
		processed_pixel_array = np.clip(pixel_array, np.percentile(pixel_array, 10), np.percentile(pixel_array, 99))
		# We convert the pixel array to store uint8 values instead of uint16, therefore saving space
		processed_pixel_array *= (255.0 / processed_pixel_array.max())
		processed_pixel_array = processed_pixel_array.astype(np.uint8)
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
		meta_file = open(os.path.join(self.root_folder, patient_folder, "meta.txt"), 'r')
		file_content = meta_file.read().split()[-1]
		meta_file.close()
		skip = file_content in ("adult_m_sport", "adult_f_sport", "U18_f", "U18_m")  # if any of these are true then we can't check hypertrophy
		hypertrophic = file_content != "Normal"
		return skip, hypertrophic

	def _update_data_holder(self, dcm_file, pixel_data, negative, chamber_view_type, heart_phase):
		if heart_phase == HeartPhaseEnum.NOTHING: return  # If it's not a systole or diastole, we don't want to save it
		systole = heart_phase == HeartPhaseEnum.SYSTOLE  # Just to shorten the code a little

		# A bit of an ugly solution, but I would have had to rewrite the program otherwise. This'll do
		image_orientation_patient = np.asarray(dcm_file.ImageOrientationPatient).reshape(2, 3)
		image_orientation_normal = np.cross(image_orientation_patient[0], image_orientation_patient[1])

		# Ugly, but it do be like that sometimes
		curr_view = 0
		if chamber_view_type == ChamberEnum.CH2:
			if heart_phase == HeartPhaseEnum.SYSTOLE:
				curr_view = self.data_holder.ch2_systole
			elif heart_phase == HeartPhaseEnum.DIASTOLE:
				curr_view = self.data_holder.ch2_diastole
			elif heart_phase == HeartPhaseEnum.EXTRA:
				self.data_holder.ch2_extra.append(dict())
				curr_view = self.data_holder.ch2_extra[-1]
		elif chamber_view_type == ChamberEnum.CH4:
			if heart_phase == HeartPhaseEnum.SYSTOLE:
				curr_view = self.data_holder.ch4_systole
			elif heart_phase == HeartPhaseEnum.DIASTOLE:
				curr_view = self.data_holder.ch4_diastole
			elif heart_phase == HeartPhaseEnum.EXTRA:
				self.data_holder.ch4_extra.append(dict())
				curr_view = self.data_holder.ch4_extra[-1]
		elif chamber_view_type == ChamberEnum.LVOT:
			if heart_phase == HeartPhaseEnum.SYSTOLE:
				curr_view = self.data_holder.lvot_systole
			elif heart_phase == HeartPhaseEnum.DIASTOLE:
				curr_view = self.data_holder.lvot_diastole
			elif heart_phase == HeartPhaseEnum.EXTRA:
				self.data_holder.lvot_extra.append(dict())
				curr_view = self.data_holder.lvot_extra[-1]

		curr_view["pixel_data"] = pixel_data
		curr_view["image_orientation"] = image_orientation_patient
		curr_view["normal"] = image_orientation_normal
		curr_view["is_normal_inverted"] = negative

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
