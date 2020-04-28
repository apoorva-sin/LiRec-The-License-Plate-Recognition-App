import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

characters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def import_data(train_dir):
    image_data = []
    target_data = []
    for each_letter in characters:
        for each in range(10):
            image_path = os.path.join(train_dir, each_letter, each_letter + '_' + str(each) + '.jpg')

            # Import image and convert to binary 
            img_details = imread(image_path, as_gray=True)
            binary_image = img_details < threshold_otsu(img_details)
           
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):

    acc = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)

    print("Cross Validation Result for ", str(num_of_fold), "-fold")
    print(acc * 100)


current_dir = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(current_dir, 'train')

image_data, target_data = import_data(train_dir)

# Defining the model
svc_model = SVC(kernel='linear', probability=True)

# Defining cross validation
cross_validation(svc_model, 4, image_data, target_data)

# Fitting the model on the training data
svc_model.fit(image_data, target_data)

# Saving the trained model
save_dir = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
joblib.dump(svc_model, save_dir+'/svc.pkl')