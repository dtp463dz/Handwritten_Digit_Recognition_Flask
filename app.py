from flask import Flask, redirect, url_for, render_template, request
from tensorflow.keras.models import load_model
from image_processing import process_img
import os
import numpy as np

app = Flask(__name__)
#model
"""
model_path = 'mnist_model.h5'
if os.path.exists(mnist_model.h5):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model không tồn tại")
"""
# UPLOAD_FOLDER = 


@app.route('/')
def home():
    return render_template("index.html")

 
if __name__ == "__main__":
    app.run(debug=True)