"""
Cross Validation - example of using validating our model using several
k-fold cross validation techniques and optimizing it using paralleization
techniques
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD


def build_logistic_regression_model():
    '''Builds our logistic regression model'''
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(SGD(lr=0.5),
                  'binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    build_logistic_regression_model()


if __name__ == '__main__':
    main()
