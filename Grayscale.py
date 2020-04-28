from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Import the image
def grayscale(path):
    image = imread(path, as_gray=True)

    # Display image shape
    print("Image Shape: {}".format(image.shape))

    gray_image = image * 255

    # Plot grayscaled and binary images
    # fig, (axis1, axis2) = plt.subplots(1, 2)
    # axis1.imshow(gray_image, cmap="gray")
    threshold_value = threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value
    return binary_image
    # axis2.imshow(binary_image, cmap="gray")

    # plt.show()

