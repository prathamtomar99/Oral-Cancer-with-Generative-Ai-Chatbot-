from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import torch
from torchvision import transforms
import cv2
app = Flask(__name__)

upl_fol = 'static/uploads'
app.config['UPLOAD_FOLDER'] = upl_fol
os.makedirs(upl_fol, exist_ok=True)

model = tf.keras.models.load_model('./models/efficient_net_B2.keras')
# model = tf.keras.models.load_model('models/best_model_resnet.keras')
def prediction_to_name(data):
    if(data>0.4):  #threshold =0.4
        return 1
    else:
        return 0
CLASS_NAMES=["CANCER","NON CANCER"]

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '': # no file selected
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            CLASS_NAMES=["CANCER","NON CANCER"]
            test_img=cv2.imread(file_path)
            test_img= cv2.resize(test_img,(224,224))
            test_img= tf.expand_dims(test_img,axis=0)
            predicted_class=CLASS_NAMES[prediction_to_name(model.predict(test_img)[0][0])]
            print(model.predict(test_img)[0][0])
            return render_template('result.html', predicted_class=predicted_class, filename=filename)

    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
# 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