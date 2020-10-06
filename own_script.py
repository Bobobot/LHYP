from con2img import draw_contourmtcs2image as draw
from dicom_preprocessor import DicomPreProcessor

image_folder = '../hypertrophy/10635813AMR806/la'
# con_file = '../hypertrophy/10635813AMR806/la/contours.con'

preprocessor = DicomPreProcessor(image_folder)
