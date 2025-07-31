import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MarvellousPredictor():

    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    print("Value of Independent Variable",X)
    print("Value of Dependent Variable",Y)

    X_Sum = 0
    Y_Sum = 0

    X_Mean = np.mean(X)
    Y_Mean = np.mean(Y)

    print("Mean of X:",X_Mean)
    print("Mean of Y:",Y_Mean)

    n = len(X)
    print(n)
    numerator = 0
    Denomentor = 0

    for i in range(n):
        numerator = numerator + (X[i] - X_Mean)*(Y[i]-Y_Mean)
        Denomentor = Denomentor + (X[i] - X_Mean)**2

    m = numerator/Denomentor
    print("Slop of line m is:",m)

    C = (Y_Mean) - (m * X_Mean)
    print("Y intercept of line is:",C)

    x = np.linspace(1,6,n)
    y = C + m * x

    plt.plot(x,y,color = 'g',label = " Regression Line")
    plt.scatter(X,Y,color = 'r',label = " Scatter plot")

    plt.xlabel("X:Independent Variable")
    plt.ylabel("Y:Dependent Variable")

    plt.legend()
    plt.show()

def main():
    MarvellousPredictor()
if __name__ == "__main__":
    main()