from flask import Flask, request, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('C:/Users/chait/OneDrive/Documents/Sasvaat Pvt Ltd/action-detection/action-detection-model.h5')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale as per training
    return img_array

def predict_activity(img_path):
    img = load_and_preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a valid file is provided
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            predicted_class = predict_activity(file_path)
            return f'Predicted class: {predicted_class}'
    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an Image for Activity Recognition</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)