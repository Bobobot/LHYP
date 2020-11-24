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
# nnet._draw_plots([0, 0.5, 1, 2, 3, 4, 5, 5.5, 6, 7, 2.5, 6, 7.6], [0, 10, 1.5, 3, 2, 9.2])
nnet.train(15)
nnet.test()
