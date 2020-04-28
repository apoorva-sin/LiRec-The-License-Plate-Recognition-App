import os
import characters
import Grayscale
import CCA
import characters

def transform(image_path):

    transformed_image = Grayscale.grayscale(image_path)
    license_plate = CCA.cca(transformed_image)
    chars, column_list = characters.character_list(license_plate)

    return chars, column_list


