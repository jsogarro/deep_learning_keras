"""
Sample Plots - showcasing some useful matplot lib functionality
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def main():
    # generate random data
    d1 = np.random.normal(0, 0.1, 1000)
    d2 = np.random.normal(1, 0.4, 1000) + np.linspace(0, 1, 1000)
    d3 = 2 + np.random.random(1000) * np.linspace(1, 5, 1000)
    d4 = np.random.normal(3, 0.2, 1000) + 0.3 * np.sin(np.linspace(0, 20, 1000))

    # stack our datasets
    data = np.vstack([d1, d2, d3, d4]).transpose()

    # create a frame from the data 
    df = pd.DataFrame(data, columns=['d1', 'd2', 'd3', 'd4'])
    print(df.head())

    # plot and show the data 
    plt.plot(df)
    plt.title('Line Graph')
    plt.legend(['d1', 'd2', 'd3', 'd4'])
    plt.show()

if __name__ == '__main__':
    main()



