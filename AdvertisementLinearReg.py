import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def AdvertisingCaseStudy(Datapath):
    df = pd.read_csv(Datapath)
    print("Dataset Samples:")
    print(df.head())

    print("Updated Dataset")
    df.drop(columns = ['Unnamed: 0'], inplace = True)
    print(df.head())

    print("Missing Value of Each Column:",df.isnull().sum())

    print("Statistical Summary:",df.describe())

    print("Correlation matrix")
    print(df.corr())

    plt.figure(figsize = (10,5))

    sns.heatmap(df.corr(),annot = True,cmap = 'coolwarm')
    plt.title("Advertisement Correlation Heatmap")
    plt.show()

    sns.pairplot(df)
    plt.suptitle("Pairplot of feture",y = 1.02)
    plt.show()

    x = df[['TV','radio','newspaper']]
    y = df[['sales']]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
    model = LinearRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    MSE = metrics.mean_squared_error(y_test,y_pred)

    RMSE = np.sqrt(MSE)

    r2 = metrics.r2_score(y_test,y_pred)

    print("Mean square error",MSE)
    print("Root Mean Square Error",RMSE)
    print("R2 square:",r2)

    print("Model Coefficient are:")

    for col,coef in zip(x.columns,model.coef_):
        print(f"{col}:{coef}")

    print("Y Intercept:",model.intercept_)

    plt.figure(figsize =(8,5))

    plt.scatter(y_test,y_pred,color = 'blue')

    plt.xlabel("Actual Sales")

    plt.ylabel("Predicted sale")

    plt.title(" Advertisement case study")

    plt.grid(True)

    plt.show()

def main():
    AdvertisingCaseStudy("Advertising.csv")

if __name__ == "__main__":
    main()