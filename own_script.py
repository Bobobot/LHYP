# from con2img import draw_contourmtcs2image as draw
# from dicom_preprocessor_old import DicomPreProcessorOld

from NNetwork import CNN, NNetworkHelper
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

# reader = TestReader("6211434AMR801.rick")
# reader.load()

# CNN training test

nnet = NNetworkHelper("data")
nnet.train(25)
nnet.test()
