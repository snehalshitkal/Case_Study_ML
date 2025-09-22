################################################################################################
#               Rerquired Python Packages
################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


def MarvellousHeadBrainLinear(Datapath):

    Line = "*"*50
    df = pd.read_csv(Datapath)

    print("First few records of the dataset are:")
    print(Line)
    print(df.head())
    print(Line)

    print("Statistical information of the dataset")
    print(Line)
    print(df.describe())
    print(Line)

    x = df[['Head Size(cm^3)']]
    y = df[['Brain Weight(grams)']]

    print("Independent variable are:Head Size")

    print("Dependent variable are:Brain Weight")

    print("Total record in dataset:",x.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

    print("Dimension of trainig dataset",x_train.shape)
    print("Dimension of Testing Dataset",y_test.shape)

    model = LinearRegression()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test,y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test,y_pred) 
    
    
    print("Visual Representation")

    plt.figure(figsize = (8,5))

    plt.scatter(x_test,y_test,color = 'blue',label = 'Actual')

    plt.plot(x_test.values.flatten(),y_pred,color = 'red',linewidth = 2,label= "Regression")

    plt.xlabel('Head Size(cm^3)')
    plt.ylabel('Brain  Weight(grams)')

    plt.title("Marvellous Head brain Regression")

    plt.legend()
    plt.grid(True)
    plt.show()

    print("Result of Case study")

    print("Slop of Line(m)",model.coef_[0])

    print("Intercept (C):",model.Intercept_)

    print("Mean Squared Error is",mse)

    print("Root Mean Squared Error is",rmse)

    print("R square Value",r2)

    
################################################################################################
'''
    Functioin Name  :  main
    Description     :  Main Function from where execution start
    Author          :  Snehal Rohit Shitkal
'''
################################################################################################

def main():
    MarvellousHeadBrainLinear("MarvellousHeadBrain.csv")

################################################################################################
'''
            Application Starter
'''
################################################################################################

if __name__ == "__main__":
    main()