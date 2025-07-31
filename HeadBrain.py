import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

import matplotlib.pyplot as plt

def MarvellousHeadBrainLinear(Datapath):

    Line = "*"*50
    df = pd.read_csv(Datapath)

    x = df[['Head Size(cm^3)']]
    y = df[['Brain Weight(grams)']]

    print("Independent variable  Head size:")
    print("Dependent variable of Brain weight:")

    print("Total record in Dataset",x.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
    
    print("Dimension of Training datset ",x_train.shape)
    print("Dimension of testing dataset",y_test.shape)

    model = LinearRegression()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_pred,y_test)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_pred,y_test)

    
    print("Visual Representation:")

    plt.figure(figsize = (8,5))

    plt.scatter(x_test,y_test,color = 'blue',label = 'Actual')

    plt.plot(x_test.values.flatten(),y_pred,color = 'red',linewidth = 2,label = "Regression")
    
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weight(grams)")

    plt.title("Marvellous Head brain Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Result of Case Study:")

    print("Slop of Line(m)",model.coef_[0])

    print("Model of Intercept(C)",model.intercept_)
    print("Mean square error",mse)

    print("Root mean square error",rmse)
    print("R square value",r2)

def main():
    MarvellousHeadBrainLinear("MarvellousHeadBrain.csv")
if __name__ =="__main__":
    main()