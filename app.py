from flask import Flask, redirect, url_for, render_template, request, jsonify
from tensorflow.keras.models import load_model
from image_processing import process_img
import numpy as np
import os

app = Flask(__name__)
model_path = 'mnist_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model không tồn tại")

UPLOAD_FOLDER = 'uploads/'
# check folder upload
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predic_digit(digit_img):
    # chuẩn hóa ảnh số
    digit_img = digit_img.astype('float32') / 255.0
    # thêm chiều batch vào ảnh
    digit_img = np.expand_dims(digit_img, axis=-1)
    digit_img = np.expand_dims(digit_img, axis=0)
    # Dự đoán số từ mô hình
    predict = model.predict(digit_img)
    return np.argmax(predict)

@app.route('/')
def home():
    return render_template("index.html", predict=None)

@app.route('/results', methods=['POST'])
def results():
    # kiểm tra có tệp tin được gửi lên
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})
    file = request.files['file']
    
    # kiểm tra người dùng chọn tệp
    if file.filename == '':
        return jsonify({'error:' 'No select file'})

    # lưu tệp vào thư mục upload
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        #Tiền xử lý ảnh và dự đoán các chữ số
        digits, _, _ = process_img(file_path)
        if not digits:
            return jsonify({'error': 'No digits found'})

        predictions = [predic_digit(digit) for digit in digits]
        res = ''.join(map(str, predictions))
        # chuyển kết quả thành 1 chuỗi
        res = str(res)
        return jsonify({'predict': res})

        

if __name__ == "__main__":
    app.run(debug=True)