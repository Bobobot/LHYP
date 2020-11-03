import os
import pickle
import numpy as np

import cv2


class TestReader:
	def __init__(self, filename):
		self._filename = filename

	def load(self):
		with open(os.path.join("data", self._filename), "rb") as file:
			loaded_pickle = pickle.load(file)
			print(loaded_pickle)
			cv2.namedWindow('2 channel diastole', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('2 channel diastole', 600, 600)
			cv2.imshow('2 channel diastole', loaded_pickle.ch2_diastole["pixel_data"])
			cv2.waitKey()
