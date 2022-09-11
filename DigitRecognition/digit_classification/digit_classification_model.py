import numpy as np
from tensorflow import keras as tfk
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

def plot_history(fit_history):
    """
    Create plot showing accuracy and loss per epoch of model fitting

    Arg:
        fit_history (history object): History information of fitted Keras model
    """
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.subplot(1, 2, 1)
    plt.plot(fit_history.history["categorical_accuracy"], label="train data")
    plt.plot(fit_history.history["val_categorical_accuracy"], label="test data")
    plt.title("Accuracy history")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(fit_history.history["loss"], label="train data")
    plt.plot(fit_history.history["val_loss"], label="test data")
    plt.title("Loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Load image data from tensorflow package
    (X_train, y_train), (X_test, y_test) = tfk.datasets.mnist.load_data()

    # Shuffle to ensure that no dependencies in image / label
    # order exist in the image data set upon model training.
    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    y_train = y_train[shuffler]

    # Keep categorical data for later model evaluation
    y_test_true = y_test.copy()

    # OneHotEncode labels
    y_train = tfk.utils.to_categorical(y_train)
    y_test = tfk.utils.to_categorical(y_test)

    # Clear session
    tfk.backend.clear_session()

    # Create CNN model
    model = tfk.models.Sequential([

        # First convolutional layer
        Conv2D(filters=16,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1)),
        # Pooling by maximum value
        MaxPooling2D(pool_size=(2,2),
                    strides=(2,2),
                    padding="same"),
        # Normalization
        BatchNormalization(),

        # Second convolutional layer
        Conv2D(filters=32,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"),
        # Pooling by maximum value
        MaxPooling2D(pool_size=(2,2),
                    strides=(2,2),
                    padding="same"),

        # Flatten array
        Flatten(),
    
        # Fully connected
        # layer 1
        Dense(150, activation="relu"),
        Dropout(0.2),

        # layer 2
        Dense(50, activation="relu"),
        Dropout(0.2),

        # Output layer
        Dense(10, activation="softmax")
    ])

    print(model.summary())

    # Compile the model
    model.compile(optimizer=tfk.optimizers.Adam(), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    # Train the model
    fit_history = model.fit(X_train, y_train, batch_size=50, epochs=15, validation_split=0.2)

    # Plot history of accuracy and loss upon modelfitting for inspection
    plot_history(fit_history)

    # Model evaluation

    # Make prediction for test set
    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)

    # Determine accuracy measures for test prediction
    print("Prediction accuracy for test set:", round(accuracy_score(y_test_true, y_test_pred),3))

    # Create confusion matrix
    plt.close()
    plt.rcParams["figure.figsize"] = (8, 8)
    cm = confusion_matrix(y_true=y_test_true, y_pred=y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_true))
    disp.plot()
    plt.show()

    # Save model
    model.save('digit_classification_model.h5')