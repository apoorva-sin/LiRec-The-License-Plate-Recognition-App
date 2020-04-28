from skimage import measure
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def cca(transformed_image):
    # Group the connected regions together
    label_image = measure.label(transformed_image)

    # Determine the dimensions of the licence plate
    min_height_fraction = 0.03
    max_height_fraction = 0.2
    min_width_fraction = 0.15
    max_width_fraction = 0.4

    # Set plate dimensions
    plate_dimensions = (min_height_fraction*label_image.shape[0], max_height_fraction*label_image.shape[0],
                        min_width_fraction*label_image.shape[1], max_width_fraction*label_image.shape[1])

    min_height, max_height, min_width, max_width = plate_dimensions

    candidates_coordinates = []
    candidates = []
    # fig, (axis1) = plt.subplots(1)
    # axis1.imshow(gray_image, cmap="gray")


    for region in regionprops(label_image):
        # Remove possible candidates with too small an area
        if region.area < 50:
            continue

        # Get the size of the candidate region
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            candidates.append(transformed_image[min_row:max_row,
                                    min_col:max_col])
            candidates_coordinates.append((min_row, min_col,
                                                max_row, max_col))
            
            # Mark the candidates with a rectangular border
            # rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="blue", linewidth=2, fill=False)
            # axis1.add_patch(rectBorder)

    # Pick out the license plate from possible candidates
    highest_average = 0
    license_plate = []

    for each_candidate in candidates:
        backup = each_candidate
        height, width = each_candidate.shape
        
        threshold_value = threshold_otsu(each_candidate)
        each_candidate = each_candidate < threshold_value

        total_white_pixels = 0
        for column in range(width):
            total_white_pixels += sum(each_candidate[:, column])
        
        average = float(total_white_pixels) / width
        if average >= highest_average:
            license_plate = backup

    return license_plate

# plt.show()