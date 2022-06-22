import numpy as np
import pandas as pd
from munz_l_test import train_predict
from sklearn.metrics import accuracy_score
 

def main():
    randomSeed = 420
    np.random.seed(randomSeed)

    X_train_data = pd.read_csv("X_train.csv", index_col=0).values
    y_train_data = pd.read_csv("y_train.csv", index_col=0).values

    #X_test = np.zeros((2,1875))

    n_train = int(0.95 * len(X_train_data))
    print(n_train)

    X_train = X_train_data[:n_train]  # starting from one until traing -1
    X_test = X_train_data[n_train:]
    y_train = y_train_data[:n_train]
    y_test = y_train_data[n_train:]
    print(X_test.shape)


    y_pred = train_predict(X_train, y_train, X_test)
    print("THIS IS THE SHAPE!: ")
    print(y_pred.shape)

    print("Score: " + str(accuracy_score(y_test, y_pred)))


if __name__ == "__main__":
    main()