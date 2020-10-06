import pickle

from dicom_data_holder import DicomDataHolder
from utils import get_logger
import pydicom as dicom
import numpy as np
import os

logger = get_logger(__name__)

class DicomPreProcessor:

	def __init__(self, folder_name):
		dcm_files = sorted(os.listdir(folder_name))

		# we only care about files that have a filename longer than 4 characters
		dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
		if len(dcm_files) == 0:  # sometimes the order number is missing at the end
			dcm_files = sorted(os.listdir(folder_name))

		for file in dcm_files:
			try:
				temp_ds = dicom.dcmread(os.path.join(folder_name, file))
				file_name = os.path.splitext(file)[0]
				self.dumpToPickle(temp_ds, folder_name, file_name)
				print('debug')
			except Exception as e:
				self.broken = True
				logger.error('Exception while reading dcm file.' + repr(e))
				print('Hülye logger nem működik de exception volt' + repr(e))
				return

	def dumpToPickle(self, temp_ds, folder_name, file_name):
		dicom_data_holder = DicomDataHolder()
		dicom_data_holder.imageOrientationPatient = temp_ds.ImageOrientationPatient
		dicom_data_holder.imagePositionPatient = temp_ds.ImagePositionPatient
		dicom_data_holder.HeartRate = temp_ds.HeartRate
		dicom_data_holder.pixel_array = temp_ds.pixel_array
		# TODO: log which files you've already finished with
		file = open(file_name, 'wb')
		pickle.dump(dicom_data_holder, file)
		file.close()
