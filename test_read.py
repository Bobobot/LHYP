import os
import pickle
import numpy as np

import cv2


class TestReader:
	def __init__(self, filename):
		self._filename = filename
		self.win_name = 'test_read'

	#ugly test method pls ignore
	def load(self):
		with open(os.path.join("data", self._filename), "rb") as file:
			loaded_pickle = pickle.load(file)
			print(loaded_pickle)

			cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.win_name, 600, 600)

			self.show_img('2 chamber systole', loaded_pickle.ch2_systole)
			for i in range(len(loaded_pickle.ch2_extra)):
				self.show_img(f'{i+1}. ch2 extra', loaded_pickle.ch2_extra[i])
			self.show_img('2 chamber diastole', loaded_pickle.ch2_diastole)
			self.show_img('4 chamber systole', loaded_pickle.ch4_systole)
			for i in range(len(loaded_pickle.ch4_extra)):
				self.show_img(f'{i+1}. ch4 extra', loaded_pickle.ch4_extra[i])
			self.show_img('4 chamber diastole', loaded_pickle.ch4_diastole)
			self.show_img('lvot diastole', loaded_pickle.lvot_systole)
			for i in range(len(loaded_pickle.lvot_extra)):
				self.show_img(f'{i+1}. lvot extra', loaded_pickle.lvot_extra[i])
			self.show_img('lvot diastole', loaded_pickle.lvot_diastole)

	def show_img(self, new_win_name, pickle_dict):
		cv2.setWindowTitle(self.win_name, new_win_name)
		cv2.imshow(self.win_name, pickle_dict["pixel_data"])
		cv2.waitKey()
