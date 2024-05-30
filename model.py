import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Tải bộ dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chuẩn bị dữ liệu: chia tỷ lệ giá trị pixel thành [0, 1] và chuyển thành định dạng phù hợp cho mô hình
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.0
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
model.fit(X_train, y_train, epochs=20, batch_size=200, validation_data=(X_test, y_test_cat))

# Lưu mô hình
model.save('mnist_model.h5') 

# Dự đoán trên tập kiểm tra
y_pred_cat = model.predict(X_test)
y_pred = np.argmax(y_pred_cat, axis=1)

# Hàm tính toán các chỉ số đánh giá
def calculate_metrics(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    
    # Tính Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # Tính Precision, Recall, F1 Score cho từng lớp
    precision_list = []
    recall_list = []
    f1_list = []
    
    for class_id in np.unique(y_true):
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Tính giá trị trung bình của Precision, Recall, F1 Score
    precision_macro = np.mean(precision_list)
    recall_macro = np.mean(recall_list)
    f1_macro = np.mean(f1_list)
    
    return accuracy, precision_macro, recall_macro, f1_macro

# Tính toán các chỉ số đánh giá
accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

# In ra các chỉ số đánh giá
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Lưu các chỉ số đánh giá vào tệp CSV
metrics = {
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('evaluation_metrics.csv', index=False)
