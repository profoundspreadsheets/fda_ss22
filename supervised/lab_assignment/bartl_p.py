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
    """
    This method is only relevant for my own model validation.
    It will not be necessary to call during the final implementation, as
    train and test set will be supplied separately.
    """
    splitat = int(percentage*len(X_train))

    X_train_split = X_train[:splitat]
    X_test_split = X_train[splitat:]
    y_train_split = y_train[:splitat]
    y_test_split = y_train[splitat:]

    return X_train_split, X_test_split, y_train_split, y_test_split

def buildModel():
    """
    I knew from CMKE course at the university that convolutional neural networks
    are best suited for image classification problems. So I knew I was going to 
    build a CNN for this assignment. 

    The model is influenced by the following
    https://colab.research.google.com/github/AviatorMoser/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb?pli=1#scrollTo=V36GnqLfNq-W

    The dataset is similar, but I still wanted to implement some of my own experimentation.
    I did use muliple convolational layers, however I decided for a bigger kernel in the first
    layer to improve the performance a bit. Unfortunately I did not have access to my Cuda capable
    GPU so I had to learn on my CPU, which is why I opted for some performance gains over accuracy.

    The data has to be reshaped which is done in the train_predict method. It is reflattened before
    the last hidden layer.
    """
    nn_outputs = 26
    model = Sequential(
        [
            # Conv layer
            Conv2D(32, (5, 5), input_shape=(25,25,3), activation='relu'), 
            # Normalization
            BatchNormalization(),
            # Pooling for some quicker learning, downsamples picture
            MaxPooling2D(pool_size=(2,2)),

            # Deeper conv layer, should capture combined patters from prev layer
            Conv2D(64, (3, 3), activation='relu'), 
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
            
            # We increase the filters in the conv layers, the raw input could be 
            # influenced by noise, so we use the first simple conv layers to
            # extract some information and increase the complexity to hopefully
            # work with cleaner data than the input data
            Conv2D(128, (3, 3), activation='relu'), 
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
            
            # Flattening (make 1-dimensional vector)
            Flatten(),

	        # Idk how effective this is, but it helps grow the score	
            Dense(units=256, activation='relu'),
            
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
    
    ## Shuffle Data
    shuffled = np.arange(len(X_train))
    np.random.shuffle(shuffled)

    X_train = X_train[shuffled]
    y_train= y_train[shuffled]

    ## Split in training and test
    ## In the final implementation we should not split the dataset
    ## Comment this out
    ## TODO if I forget to comment out please be so kind and do so :)
    #X_train, X_test, y_train, y_test = splitDataset(0.8, X_train, y_train)

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
    model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=128)


    ## Make prediction
    y_pred = np.argmax(model.predict(X_test), axis=1)

    ### Debug
    #print ("Score: " + str(accuracy_score(y_test, y_pred)))


    # --------------------------

    # check that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),)

    return y_pred


