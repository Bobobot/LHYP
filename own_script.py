# from con2img import draw_contourmtcs2image as draw
# from dicom_preprocessor_old import DicomPreProcessorOld
import torch
from torch import nn

from NNetwork import CNN
from test_read import TestReader
from test_rotation import TestRotation
from vector_average_calc import VectorAverageCalc
from dicom_preprocessor import DicomPreprocessor

image_folder = '../hypertrophy'

# testRot = TestRotation()
# testRot.rot2()

# vec_avg = VectorAverageCalc()

# preprocessor = DicomPreprocessor(image_folder)
# preprocessor.process()

reader = TestReader("10022015AMR806.rick")
reader.load()

# CNN training test

