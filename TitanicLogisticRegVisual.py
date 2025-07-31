import numpy as np
import pandas as pd

from matplotlib.pyplot import figure,show
import matplotlib.pyplot as plt
import seaborn as sns

from seaborn import countplot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix

def TitanicLogisticRegression(Datapath):

    df = pd.read_csv(Datapath)
    print("Titanic Dataset Loaded Successfully:")
    print(df.head())

    print("Dimention of Dataset:",df.shape)

    df.drop(columns = ['Passengerid','zero'],inplace = True)

    print("Dimensional of dataset :",df.shape)

    df['Embarked'].fillna(df['Embarked'].mode()[0],inplace = True)
    
    figure()
    target = "Survived"
    countplot(data = df,x = target).set_title("Survived or not Survived")
    show()

    figure()
    target = "Survived"
    countplot(data = df , x = target,hue = 'Sex').set_title("Based on Gender Survived")
    show()

    figure()
    target = "Survived"
    countplot(data = df , x = target,hue = 'Sex').set_title("Based on PClass")
    show()

    figure()
    df['Age'].plot.hist().set_title("Age Report")
    show()

    figure()
    df['Fare'].plot.hist().set_title("Fare Report")
    show()

    plt.figure(figsize = (10,6))
    sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
    plt.title("Feture Correlation HeatMap")
    plt.show()

    x = df.drop(columns = ['Survived'])
    y = df['Survived']

    print("Dimention of Target",x.shape)
    print("Dimention of Labels",y.shape)

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train,x_test,y_train,y_test = train_test_split(x_scale,y,test_size = 0.2 , random_state = 42)

    model = LogisticRegression()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)

    confuseMatrix = confusion_matrix(y_test,y_pred)

    print("Accuracy:",accuracy)

    print("Confusion Matrix")

    print(confuseMatrix)

def main():
    TitanicLogisticRegression("TitanicDataset.csv")
if __name__ =="__main__":
    main()