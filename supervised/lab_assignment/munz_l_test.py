import numpy as np  # essential for everything
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


# define additional functions here


def train_predict(X_train, y_train, X_test):

    # check that the input has the correct shape
    #assert X_train.shape == (77220, 1875)
    #assert y_train.shape == (77220, 1)

    # to test your implementation, the test set should have a shape of (n_test_samples, 1875)
    assert len(X_test.shape) == 2
    assert X_test.shape[1] == 1875

    # --------------------------
    # add your data preprocessing, model definition, training and prediction between these lines

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform the data with the fitted scalar
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # Reshaping the data flattened data again since we are working with 25x25x3 images
    X_train = X_train.reshape(len(X_train), 25, 25, 3)
    X_test = X_test.reshape(len(X_test), 25, 25, 3)

    # One-Hot Encoding of the output labels as already discussed in the lecture
    y_train_cat = to_categorical(y_train)


    ## MODEL ##

    model = Sequential()

    # Inputting the Data with an input shape of the reshaped data 25x25x3
    model.add(Conv2D(32, (3, 3), input_shape=(25, 25, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # First hidden layer and MaxPooling2D
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # Second hidden layer and MaxPooling2D as well as flattening
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())  # Flatten the data again for the final layers

    # Output layer with 26 nodes according to the shape of our data and softmax activation function
    model.add(Dense(26))
    model.add(Activation('softmax'))

    # Printing the summary of the model
    model.summary()

    # Compiling the model itself
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

    # Fitting the model to the provided training data
    model.fit(X_train, y_train_cat, epochs=10, validation_split=0.01, batch_size=16)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    # --------------------------

    # check that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),)

    return y_pred
