import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Tải bộ dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chuẩn bị dữ liệu: chia tỷ lệ giá trị pixel thành [0, 1] và chuyển thành định dạng phù hợp cho mô hình
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
X_train, X_test = X_train / 255.0, X_test / 255.0
# Chuyển nhãn thành dạng one-hot encoding
y_train = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Tạo và huấn luyện mô hình
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=20, validation_data=(X_test, y_test_cat))

# Lưu mô hình
model.save('mnist_model.h5') 

# Dự đoán trên tập kiểm tra
y_pred_cat = model.predict(X_test)
y_pred = np.argmax(y_pred_cat, axis=1)

# Hàm tính toán các chỉ số đánh giá
def calculate_metrics(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    
    metrics = []
    
    for class_id in np.unique(y_true):
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        tn = np.sum((y_true != class_id) & (y_pred != class_id))

        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics.append({
            'Class': class_id,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    return metrics

# Huấn luyện mô hình và lưu chỉ số đánh giá cho từng epoch
num_epochs = 20
batch_size = 20
all_metrics = []

for epoch in range(num_epochs):
    model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test_cat), verbose=1)
    
    # Dự đoán trên tập kiểm tra
    y_pred_cat = model.predict(X_test)
    y_pred = np.argmax(y_pred_cat, axis=1)
    
    # Tính toán các chỉ số đánh giá
    metrics = calculate_metrics(y_test, y_pred)
    
    # Thêm thông tin epoch vào chỉ số đánh giá
    for metric in metrics:
        metric['Epoch'] = epoch + 1
        all_metrics.append(metric)

# Chuyển đổi danh sách tất cả các chỉ số đánh giá thành DataFrame
metrics_df = pd.DataFrame(all_metrics)

# Lưu DataFrame vào tệp CSV
metrics_df.to_csv('evaluation_metrics.csv', index=False)

# In ra các chỉ số đánh giá
print(metrics_df)

"""

import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger

# Tải dữ liệu
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Tiền xử lý dữ liệu
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Khởi tạo mô hình
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Đường dẫn lưu file log
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, "training_log.csv")

# Khởi tạo CSVLogger callback
csv_logger = CSVLogger(log_file_path, append=False)

# Huấn luyện mô hình
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200, callbacks=[csv_logger])

# Lưu mô hình
model.save('mnist_model.h5')

"""