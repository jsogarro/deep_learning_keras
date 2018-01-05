"""
Cross Validation - example of using validating our model k-fold cross validation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

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
    # import our data
    df = pd.read_csv('../data/user_visit_duration.csv')

    # build model, inputs and outputs
    X = df[['Time (min)']].values
    y = df['Buy'].values

    # train and cross validate our model using k-fold validation with 3 partitions
    model = KerasClassifier(build_fn=build_logistic_regression_model, epochs=50)
    cv = KFold(3, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv)
    print('Cross Validation Scores: {}'.format(scores))
    print('Average Score: {}'.format(np.average(scores)))

if __name__ == '__main__':
    main()
