"""
Linear Regression with Keras -- linear regression performed using Keras
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD


def main():
    # import our data
    df = pd.read_csv('../data/weight-height.csv')

    def linear_func(x, w=0, b=0):
        '''Linear function for our regression line'''
        return x * w + b

    # calculate x
    x = np.linspace(55, 80, 100)

    # calculate y-hat
    y_hat = linear_func(x, w=0, b=0)

    # grab or independent and dependent variables
    X = df[['Height']].values
    y_actual = df['Weight'].values

    # create a sequential model with a dense layer
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))

    # get the model summary
    model.summary()

    # configure and train the model
    model.compile(Adam(lr=0.8), 'mean_squared_error')
    model.fit(X, y_actual, epochs=100)

    y_pred = model.predict(X)

    # split our data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_actual,
                                                        test_size=0.2)

    # get the model weights and biases and reset them
    W, B = model.get_weights()
    W[0, 0] = 0.0
    B[0] = 0.0
    model.set_weights((W, B))

    # re-train the model
    model.fit(X_train, y_train, epochs=100)

    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()

    y_train_mse = mse(y_train, y_train_pred)
    y_test_mse = mse(y_test, y_test_pred)

    # plot our data and regression line
    df.plot(kind='scatter',
            x='Height',
            y='Weight',
            title='Height x Weight Distribution of Adults')
    plt.plot(X, y_pred, color='red')
    plt.show()

if __name__ == '__main__':
    main()
