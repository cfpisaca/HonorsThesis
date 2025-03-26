import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

def load_data():
    """
    Load processed keypoint data from 'processed_train.pkl'.
    That file should contain (X, y, label_encoder),
    where X is shape (num_samples, 63) => (21 landmarks x 3 coords).
    """
    with open("processed_train.pkl", "rb") as f:
        X, y, label_encoder = pickle.load(f)
    return X, y, label_encoder

# 1. Load the data (Mediapipe keypoints) from 'processed_train.pkl'
X, y, train_le = load_data()
print("Original X shape:", X.shape)  # (num_samples, 63)

# 2. Reshape X: (N, 63) -> (N, 21, 3), then expand dims -> (N, 21, 3, 1)
X = X.reshape(-1, 21, 3)
X = np.expand_dims(X, axis=-1)  # shape: (N, 21, 3, 1)
print("Reshaped X for CNN:", X.shape)

# 3. Check labels & classes
num_classes = len(train_le.classes_)
print("Number of classes:", num_classes)
print("Classes:", train_le.classes_)

# 4. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

# 5. Build a CNN that won't shrink the 3-dimension to 0
#    Use (3,3) conv with padding='same' and
#    MaxPooling2D with pool_size=(2,1) so only the "landmark dimension" is halved.

model = Sequential([
    # Convolution #1: preserve the shape with same padding
    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(21, 3, 1)),
    # Pool over the "landmark dimension" only
    MaxPooling2D(pool_size=(2, 1)),

    # Convolution #2: again use same padding
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 1)),

    # Conv2D(128, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D(pool_size=(2, 1)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 6. Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Train
epochs = 10
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val),
    batch_size=32
)

# 8. Evaluate
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# 9. Save
os.makedirs("model", exist_ok=True)
model.save("model/asl_cnn_model.h5")
print("CNN model (using keypoints) saved to 'model/asl_cnn_model.h5'")

