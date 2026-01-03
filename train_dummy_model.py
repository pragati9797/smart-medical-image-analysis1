import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Example: 3 diseases
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.save("models/xray_mri_model.h5")
print("✅ Model saved successfully")
