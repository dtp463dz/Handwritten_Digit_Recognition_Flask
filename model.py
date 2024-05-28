from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Flatten, Dense

# Tải bộ dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chuẩn bị dữ liệu: chia tỷ lệ giá trị pixel thành [0, 1] và chuyển thành định dạng phù hợp cho mô hình
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.0

# Chuyển nhãn thành dạng one-hot encoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save('mnist_model.h5')
