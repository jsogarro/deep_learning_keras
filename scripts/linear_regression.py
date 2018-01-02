"""
Linear Regression by Hand - linear regression cost function example 
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def linear_func(x, w=0, b=0):
    '''Linear function for our regression line'''
    return x * w + b

def mean_squared_error(y_actual, y_pred):
    '''calculates the average of the squared errores'''
    s = (y_actual - y_pred) ** 2
    return s.mean()

def main():
    df = pd.read_csv('../data/weight-height.csv')
    print('--------DF HEAD--------')
    print(df.head())

    # calculate x
    x = np.linspace(55, 80, 100)

    # calculate y-hat
    y_hat = linear_func(x, w=0, b=0)

    # grab or independent and dependent variables
    X = df[['Height']].values
    y_actual = df['Weight'].values

    # get our predicted values based on our model
    y_pred = linear_func(X)

    result = mean_squared_error(y_actual, y_pred)
    print('--------RESULT--------')
    print(result)

    # view the change in our regression line for different biases
    bbs = np.array([-100, -50, 0, 50, 100, 150])
    mses = []
    for b in bbs:
        y_pred = linear_func(X, w=2, b=b)
        mse = mean_squared_error(y_actual, y_pred)
        mses.append(mse)
        plt.plot(X, y_pred)

    # plot the cost function
    ax = plt.subplot(122)
    plt.plot(bbs, mses, 'o-')
    plt.title('Cost as a function of b')
    plt.xlabel('b')

    plt.show()


if __name__ == '__main__':
    main()