from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def MarvellousDecisionTreeClassifier():
    iris  =load_iris()
    data = iris.data
    target = iris.target

    X_train,X_test,Y_train,Y_test = train_test_split(data,target,test_size = 0.2)

    model = tree.DecisionTreeClassifier()

    model.fit(X_train,Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_pred,Y_test)

    return accuracy

def MarvellousCalculateKNNAccuracy():
    iris = load_iris()
    data = iris.data
    target = iris.target

    X_train,X_test,Y_train,Y_test = train_test_split(data,target,test_size = 0.2)

    model = KNeighborsClassifier(n_neighbors = 5)

    model.fit(X_train,Y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_pred,Y_test)

    return accuracy
def main():
    Result = MarvellousDecisionTreeClassifier()
    print("Accuracy by Dexcision Tree Classzfier:",Result)

    Res = MarvellousCalculateKNNAccuracy()
    print("Accuracy by KNN:",Res)
if __name__ == "__main__":
    main()