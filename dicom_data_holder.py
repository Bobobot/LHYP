from dataclasses import dataclass

import numpy as np

@dataclass
class DicomDataHolder:

	def __init__(self):
		self.hypertrophic: bool

		# dictionary indexes are:
		# - pixel_data: np.array
		# - image_orientation: np.array(2, 3)
		# - normal: np.array(3)
		# - is_normal_inverted: bool

		# 2 chamber
		self.ch2_systole = dict()
		self.ch2_diastole = dict()

		# 4 chamber
		self.ch4_systole = dict()
		self.ch4_diastole = dict()

		# LVOT
		self.lvot_systole = dict()
		self.lvot_diastole = dict()
