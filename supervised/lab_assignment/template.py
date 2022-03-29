# add imports here
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Conv2D, Flatten, BatchNormalization
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# define additional functions here
def splitDataset(percentage, X_train, y_train):
    splitat = int(percentage*len(X_train))

    X_train_split = X_train[:splitat]
    X_test_split = X_train[splitat:]
    y_train_split = y_train[:splitat]
    y_test_split = y_train[splitat:]

    return X_train_split, X_test_split, y_train_split, y_test_split

def buildModel():
    nn_outputs = 26
    model = Sequential(
        [
            # Conv layer
            Conv2D(32, (3, 3), input_shape=(25,25,3), activation='relu'), 
            Conv2D(32, (3, 3), activation='relu'), 
            
            # Pooling
            MaxPooling2D(pool_size=(2,2)),
            
            # Normalization
            BatchNormalization(),
            
            # Flattening (make 1-dimensional vector)
            Flatten(),

            Dense(units=104, activation='relu'),
            
            # Decision (26 output neurons, softmax decides like argmax)
            Dense(units=nn_outputs, activation='softmax'),
        ]
    )

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
    model.summary()
    return model;  


def train_predict(X_train, y_train, X_test):
    # check that the input has the correct shape
    assert X_train.shape == (77220, 1875)
    assert y_train.shape == (77220, 1)

    # to test your implementation, the test set should have a shape of (n_test_samples, 1875)
    assert len(X_test.shape) == 2
    assert X_test.shape[1] == 1875

    # --------------------------
    # add your data preprocessing, model definition, training and prediction between these lines
    
    ## Split in training and test
    ## In the final implementation we should not split the dataset
    ## Comment this out
    ## TODO if I forget to comment out please be so kind and do :)
    X_train, X_test, y_train, y_test = splitDataset(0.8, X_train, y_train)

    ## Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ## Reshape into (x, 25, 25, 3) array for cnn
    X_train = X_train.reshape(len(X_train), 25, 25, 3)
    X_test = X_test.reshape(len(X_test), 25, 25, 3)

    y_train = to_categorical(y_train)

    ## Get and train model
    model = buildModel()

    if (input("Start fit?: ".lower())):
        model.fit(X_train, y_train, epochs=6, validation_split=0.1, batch_size=128)


    ## Make prediction
    y_pred = np.argmax(model.predict(X_test), axis=1)

    ### Debug
    print ("Score: " + str(accuracy_score(y_test, y_pred)))


    # --------------------------

    # check that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),)

    return y_pred


