from con2img import draw_contourmtcs2image as draw
from dicom_preprocessor_old import DicomPreProcessorOld
from test_rotation import TestRotation
from vector_average_calc import VectorAverageCalc
from dicom_preprocessor import DicomPreprocessor

image_folder = '../hypertrophy'

# testRot = TestRotation()
# testRot.rot2()

# vec_avg = VectorAverageCalc()

preprocessor = DicomPreprocessor(image_folder)
preprocessor.process()
