## Oral Cancer advance Deep Learning with Ensemble Techniques

#Flask Application with TensorFlow, PyTorch, OpenCV, LLM model Integration

Features
File Upload:

Users can upload images through a form, securely saved using Flask and Werkzeug's secure_filename.
users can also chat with a Chat LLM Model which is integrates using HuggingFaceHub API

TensorFlow Model Loading:

TensorFlow is used for loading pre-trained models to perform image classification or other machine learning tasks.
PyTorch Integration:

PyTorch models can be used for tasks like object detection, image transformation, or any neural network-based operation.
Image Manipulation:

Uses the Pillow library (PIL) for basic image processing tasks like resizing, format conversion, and more.
Video and Image Processing:

Utilizes OpenCV for real-time video feed processing, image reading, and various computer vision tasks.
HTML Template Rendering:

Flask’s render_template method is used to render dynamic HTML pages, enabling the display of processed results to users.
Installation
To get started with the project, you need to install the necessary dependencies.


# Clone this repository
```bash
git clone <repo-url>
```

# Navigate into the project directory
```bash
cd <project-directory>
```

# Install the required libraries
```bash
pip install -r requirements.txt
```

If a requirements.txt file is not available, you can install them individually:

```bash
pip install Flask Werkzeug tensorflow Pillow torch torchvision opencv-python
```
Usage
Run the Application:

#Start the Flask app by running:
```bash
python app.py
```

The server will run on ```http://127.0.0.1:5000/``` by default.
Upload Image:

Navigate to the home page and upload an image through the form.
View Processed Results:

File Structure
```bash
├── app.py                # Main Flask application
├── templates/            # HTML templates for rendering
│   └── index.html        # Main page template
├── static/               # Static files (CSS, JavaScript, images)
├── uploads/              # Folder for uploaded images
├── models/               # TensorFlow and PyTorch models
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```



After processing (e.g., image classification, object detection, etc.), the result will be displayed on the output page.
File Structure in inside plain.txt
# Requirements
- Python 3.x
- Flask
- TensorFlow
- PyTorch
- OpenCV
- Pillow
- License
  
# This project is licensed under the MIT License - see the LICENSE file for details.

This provides a clear and comprehensive overview of your project, helping users understand its functionality and how to get started.
