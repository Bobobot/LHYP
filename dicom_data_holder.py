from dataclasses import dataclass

import numpy as np

@dataclass
class DicomDataHolder:

	def __init__(self):
		self.hypertrophic: bool

		# 2 chamber
		self.ch2_systole: np.array
		self.ch2_diastole: np.array

		# 4 chamber
		self.ch4_systole: np.array
		self.ch4_diastole: np.array

		# LVOT
		self.lvot_systole: np.array
		self.lvot_diastole: np.array
