import pandas as pd
from sklearn.model_selection import train_test_split

def LoadData(file_path):
    df = pd.read_csv(file_path)
    print("Dataset gets loaded in memory succesfully")
    return df

def GetInformation(df):
    print("Information about thev loaded datset is : ")
    print("Shape of dataset : ",df.shape)
    print("Columns : ",df.columns)
    print("Missing values : ",df.isnull().sum())

def EncodeData(df):
    df["species"] = df["species"].map({"Setosa" : 0, "Versicolor" : 1, "Virginica" : 2})
    return df

def Split_Feture_Target(df):
    X = df.drop('species',axis = 1)
    Y = df["species"]
    return X,Y

def split(X,Y,size = 0.2):
    return train_test_split(X,Y,test_size = size)

def main():
    data = LoadData("iris.csv")
    print(data.head())   

    GetInformation(data) 

    print("Data after encoding")
    data = EncodeData(data)
    print(data.head())  

    print("Splitting feture and Label:")
    Independent,Dependent = Split_Feture_Target(data)
    print(Independent.head())
    print(Dependent.head())

    X_train,X_test,Y_train,Y_test = split(Independent,Dependent,0.3)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

if __name__  == "__main__":
    main()