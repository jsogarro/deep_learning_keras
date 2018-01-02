"""
Sample Model - super simple model using Keras + TensorFlow backend
"""
import numpy as np
from sklearn.datasets.samples_generator import make_circles

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def main():
    X, y = make_circles(n_samples=1000,
                        noise=0.,
                        factor=0.2,
                        random_state=0)

    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20)

if __name__ == '__main__':
    main()