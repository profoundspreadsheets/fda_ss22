import numpy as np
import pandas as pd
from bartl_p import train_predict

def main():
    randomSeed = 420
    np.random.seed(randomSeed)

    X_train_data = pd.read_csv("X_train.csv", index_col=0).values
    y_train_data = pd.read_csv("y_train.csv", index_col=0).values

    X_test = np.zeros((2,1875)) # spoof testset

    y_pred = train_predict(X_train_data, y_train_data, X_test)

    # You do your stuff with y_pred...

if __name__ == "__main__":
    main()
    