"""
Logistic Regression - example of using logistic regression to solve classification problems
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD


def plot_model_and_data(df, model):
    '''Plots the model input and output'''
    ax = df.plot(kind='scatter',
                 x='Time (min)',
                 y='Buy',
                 title='Actual Purchases vs. Time on Site')
    temp = np.linspace(0, 4)
    ax.plot(temp, model.predict(temp), color='green')
    plt.legend(['model output', 'model data'])
    plt.show()


def main():
    df = pd.read_csv('../data/user_visit_duration.csv')
    print(df.head())

    # set up our model
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # collect inputs and outputs
    X = df[['Time (min)']].values
    y = df['Buy'].values

    # train the model
    model.fit(X, y, epochs=50)

    # plot output
    # plot_model_and_data(df, model)

    # predict outputs and check accuracy
    y_pred = model.predict(X)
    y_class_pred = y_pred > 0.5
    print('Model Accuracy (First Run): {}'.format(accuracy_score(y, y_class_pred)))

    # check results of model on testing set and retrain the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    weights = model.get_weights()
    shaped_weights = [np.zeros(w.shape) for w in weights]
    model.set_weights(shaped_weights)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    print('Model Accuracy: {}'.format(accuracy_score(y_train, model.predict(X_train) > 0.5)))
    print('Test Accuracy: {}'.format(accuracy_score(y_test, model.predict(X_test) > 0.5)))


if __name__ == '__main__':
    main()
