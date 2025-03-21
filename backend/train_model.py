import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Load preprocessed data
def load_data():
    """
    Load processed data.
    Loads training data from processed_train.pkl.
    """
    filename = "processed_train.pkl"
    with open(filename, "rb") as f:
        X, y, label_encoder = pickle.load(f)
    return X, y, label_encoder

if __name__ == "__main__":
    X, y, train_le = load_data()

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])

    # Build a simple model
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'), # Input layer and first dense layer
        Dropout(0.5), # Sets 50% of inputs to zero, helps prevent overfitting
        Dense(64, activation='relu'), # Second dense layer
        Dropout(0.5), # Another dropout layer
        Dense(len(train_le.classes_), activation='softmax') # Softmax outputs probability distribution across all classes
    ])

    # Compile the model
    model.compile(
        optimizer='adam', # Adapts the learning rate for each parameter using Adam
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary() # Print model architecture summary

    # Train the model
    model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_val, y_val)
    )

    model.save('model/asl_model.h5') # Save the trained model
    print("Model training complete and saved to 'model/asl_model.h5'")

    # Evaluate model on the validation set
    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {test_accuracy * 100:.2f}%")
