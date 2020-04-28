import os
from sklearn.externals import joblib
import image_process

def prediction(image_path):
    # Loading the saved model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
    model = joblib.load(model_dir)

    chars, column_list = image_process.transform(image_path)
    classification_result = []

    for char in chars:
        char = char.reshape(1, -1)
        result = model.predict(char)
        classification_result.append(result)

    license_plate_characters = ''
    for eachPredict in classification_result:
        license_plate_characters += eachPredict[0]

    # Reordering the obtained plate string characters
    column_list_copy = column_list[:]
    column_list.sort()
    final_output = ''
    for each in column_list:
        final_output += license_plate_characters[column_list_copy.index(each)]

    return final_output    