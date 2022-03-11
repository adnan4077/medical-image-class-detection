from flask import Flask, jsonify, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

app.secret_key = "secret key"

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home 
@app.route('/')
def home():
    return render_template('index.html')

# Upload Image 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

# Display Image Route in template(index.html)      
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Predict 
@app.route("/predict", methods=['POST'])
def predict():
    dataset =  request.form['dataset']
    filename = request.form['filename']
    filename = app.config['UPLOAD_FOLDER']+filename
    if dataset=='derma':

        derma_labels = ['actinic keratoses and intraepithelial carcinoma', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions']
        result = predict_image_class('dm_model_2_weights', filename,'dm_model_2_weights.h5',derma_labels)
        return jsonify(result)
    else:

        retina_lables = ['0','1','2','3','4']
        result = predict_image_class('rm_model_1_with_aug_weights', filename,'rm_model_1_with_aug_weights.h5',retina_lables)
        return jsonify(result)


# load an image and predict the image class
def predict_image_class(model_name, image_location=None, model_location=None, labels=None):
    img = None
    
    # load the image from location
    img = load_img(image_location, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)

    # reshape into a single sample with 3 channels
    img = img.reshape(1, 28, 28, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    
    # load the model
    model = load_model(model_location)

    # predict the image class
    pred = model.predict(img)
    label = np.argmax(pred,axis=1)
    result = labels[label[0]]
        
    # print(f'Predicted class using model {model_name}: {result}')
    return result
