from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import torch
from torchvision import transforms
import cv2
app = Flask(__name__)
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    token="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"
)

upl_fol = 'static/uploads'
app.config['UPLOAD_FOLDER'] = upl_fol
os.makedirs(upl_fol, exist_ok=True)

with open('./models/meta_learner_logistic.pkl', 'rb') as file:
    loaded_meta_learner = pickle.load(file)

model1 = tf.keras.models.load_model('./models/efficient_net_B2.keras')
model3 = tf.keras.models.load_model('./models/vgg19.keras')
model5 = tf.keras.models.load_model('./models/best_model_resnet.keras')

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

            model1_pred= model1.predict(test_img)
            model3_pred= model3.predict(test_img)
            model5_pred= model5.predict(test_img)
            stacked_predss = np.column_stack((model1_pred, model3_pred, model5_pred))
            pred= loaded_meta_learner.predict(stacked_predss)

            predicted_class=CLASS_NAMES[prediction_to_name(pred)]
            print(pred)
            return render_template('result.html', predicted_class=predicted_class, filename=filename)

    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    try:
        # Make a text request to the Hugging Face model
        # response = client.text(user_input)
        response=""
        for message in client.chat_completion(
            messages=[{"role": "user", "content": f"{user_input}"}],
            max_tokens=100,
            stream=True,
        ):
            print(message.choices[0].delta.content, end="")
            response= response+message.choices[0].delta.content
        print(response)
        response=response.replace('\n', '<br>')
        response=response.split("**")
        final_text=""
        for i, part in enumerate(response):
            final_text +=  f"<span style='font-size: 20.5px;'>{part}</span>" if((i%2 != 0)  & (i>0)) else  part
        # print(final_text)
        return jsonify({'response': final_text})
    except Exception as e:
        print(e)
        # Handle possible errors
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)