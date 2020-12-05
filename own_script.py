from NNetwork import CNN, NNetworkHelper
from hyp_file_man import HypertrophyFileManager
from test_read import TestReader
from test_rotation import TestRotation
from vector_average_calc import VectorAverageCalc
from dicom_preprocessor import DicomPreprocessor

# image_folder = '../hypertrophy'

# testRot = TestRotation()
# testRot.rot2()

# vec_avg = VectorAverageCalc()

# preprocessor = DicomPreprocessor(image_folder)
# preprocessor.process()

# reader = TestReader("6211434AMR801.rick")
# reader.load()

# CNN training test

hyp_man = HypertrophyFileManager()
hyp_man.separate_files("data", (75, 15, 10), ("train", "validate", "test"), 0.1)


# nnet = NNetworkHelper("data")
# nnet.train(25)
# nnet.validate()
