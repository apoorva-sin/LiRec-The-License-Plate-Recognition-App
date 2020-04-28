import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def character_list(license_plate):
    plate = np.invert(license_plate)

    labelled_plate = measure.label(plate)
    print(labelled_plate.shape)

    # fig, axis1 = plt.subplots(1)
    # axis1.imshow(plate, cmap="gray")

    # Character dimensions WITHIN the license plate
    char_dim = (0.35*plate.shape[0], 0.60*plate.shape[0], 0.02*plate.shape[1], 0.15*plate.shape[1])
    min_height, max_height, min_width, max_width = char_dim

    characters = []
    counter=0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = plate[y0:y1, x0:x1]

            # Mark each character with a red rectangle border around it
            # rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
            #                             linewidth=2, fill=False)
            # axis1.add_patch(rect_border)

            # Resizing characters to make them suitable for the trained model
            # and keeping track of character order
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)
            column_list.append(x0) 

    return characters, column_list
    # plt.show()