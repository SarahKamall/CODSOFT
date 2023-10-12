from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pandas import *
import warnings
warnings.filterwarnings('ignore')

data = read_csv('IRIS.csv')
print(data.head())
print(data.info())
#then we proved that we have non null values
species = data['species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 2, 3], inplace=True) #encoding the gender
print(data.head())

X = data.drop('species', axis=1)
y = data['species']

print("The number of rows for each specie: ", y.value_counts())
#then we proved that we have 50 row from each species

def VisualizationSpecies():
    #Blue color represents the setosa, pink color represents the versicolor and the green color represents the virginica
    plt.figure('figure1')
    plt.title('Blue = Setosa, Pink = Versicolor, Green = Virginica')
    plt.scatter((data.loc[0:49, "sepal_length"]), (data.loc[0:49, "sepal_width"]), c="blue")
    plt.scatter((data.loc[50:100, "sepal_length"]), (data.loc[50:100, "sepal_width"]), c="pink")
    plt.scatter((data.loc[100:150, "sepal_length"]), (data.loc[100:150, "sepal_width"]), c="green")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.show()

    plt.figure('figure2')
    plt.title('Blue = Setosa, Pink = Versicolor, Green = Virginica')
    plt.scatter((data.loc[0:49, "sepal_length"]), (data.loc[0:49, "petal_length"]), c="blue")
    plt.scatter((data.loc[50:100, "sepal_length"]), (data.loc[50:100, "petal_length"]), c="pink")
    plt.scatter((data.loc[100:150, "sepal_length"]), (data.loc[100:150, "petal_length"]), c="green")
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.show()

    plt.figure('figure3')
    plt.title('Blue = Setosa, Pink = Versicolor, Green = Virginica')
    plt.scatter((data.loc[0:49, "sepal_length"]), (data.loc[0:49, "petal_width"]), c="blue")
    plt.scatter((data.loc[50:100, "sepal_length"]), (data.loc[50:100, "petal_width"]), c="pink")
    plt.scatter((data.loc[100:150, "sepal_length"]), (data.loc[100:150, "petal_width"]), c="green")
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Width")
    plt.show()

    plt.figure('figure4')
    plt.title('Blue = Setosa, Pink = Versicolor, Green = Virginica')
    plt.scatter((data.loc[0:49, "sepal_width"]), (data.loc[0:49, "petal_length"]), c="blue")
    plt.scatter((data.loc[50:100, "sepal_width"]), (data.loc[50:100, "petal_length"]), c="pink")
    plt.scatter((data.loc[100:150, "sepal_width"]), (data.loc[100:150, "petal_length"]), c="green")
    plt.xlabel("Sepal Width")
    plt.ylabel("Petal Length")
    plt.show()

    plt.figure('figure5')
    plt.title('Blue = Setosa, Pink = Versicolor, Green = Virginica')
    plt.scatter((data.loc[0:49, "sepal_width"]), (data.loc[0:49, "petal_width"]), c="blue")
    plt.scatter((data.loc[50:100, "sepal_width"]), (data.loc[50:100, "petal_width"]), c="pink")
    plt.scatter((data.loc[100:150, "sepal_width"]), (data.loc[100:150, "petal_width"]), c="green")
    plt.xlabel("Sepal Width")
    plt.ylabel("Petal Width")
    plt.show()

    plt.figure('figure6')
    plt.title('Blue = Setosa, Pink = Versicolor, Green = Virginica')
    plt.scatter((data.loc[0:49, "petal_length"]), (data.loc[0:49, "petal_width"]), c="blue")
    plt.scatter((data.loc[50:100, "petal_length"]), (data.loc[50:100, "petal_width"]), c="pink")
    plt.scatter((data.loc[100:150, "petal_length"]), (data.loc[100:150, "petal_width"]), c="green")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print("The number of rows for each trained specie: ", y_train.value_counts()) #the data is balanced well

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def Predict():
    # Getting input from the user to know the prediction of specie of specific flower.. It's optional.
    print('\n Enter the sepal length:')
    s_len = input()
    print("\n Enter the sepal width: ")
    s_wid = input()
    print("\n Enter the petal length: ")
    p_len = input()
    print("\n Enter the petal width: ")
    p_wid = input()
    test_row = [[s_len, s_wid, p_len, p_wid]]
    return test_row

def CalculateAccuracy(y_test, pred):
    print('\n-------------------- Testing accuracy --------------------')
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, pred)))
    print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_test, pred)))
    print('\n------------------ Confusion Matrix -----------------\n')
    print(confusion_matrix(y_test, pred))
    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, pred, average='weighted')))
    print('\n--------------- Classification Report ---------------\n')
    print(classification_report(y_test, pred))

# 1- RANDOM FOREST CLASSIFIER Model
def MyRandomForestClassifier():
    rf = RandomForestClassifier(max_depth=6, n_estimators=100)
    rclf = rf.fit(X_train_scaled,y_train)
    y_pred1 = rclf.predict(X_test_scaled)
    print('\n 1- RANDOM FOREST CLASSIFIER')
    print('\n-------------------- Training accuracy --------------------')
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y_train, rclf.predict(X_train_scaled))))
    CalculateAccuracy(y_test, y_pred1)
    while True:
        print("Do you want to know the specie of specific flower? if yes press 1, if not press 2: ")
        number = input()
        if number == '1':
            test_row = Predict()
            test_row_scaled = scaler.transform(test_row)
            y_pred = rclf.predict(test_row_scaled)
            if y_pred == 1:
                print("IRIS Setosa.")
            elif y_pred == 2:
                print("IRIS Versicolor.")
            else:
                print("IRIS Virginica.")
        elif number == '2':
            break
        else:
            print("Invalid input.")

# 2- KNeighborsClassifier Model
def KNN():
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    knnclf = neigh.fit(X_train_scaled, y_train)
    y_pred2 = knnclf.predict(X_test_scaled)
    print('\n 2- KNeighborsClassifier')
    print('\n-------------------- Training accuracy --------------------')
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y_train, knnclf.predict(X_train_scaled))))
    CalculateAccuracy(y_test, y_pred2)
    while True:
        print("Do you want to know the specie of specific flower? if yes press 1, if not press 2: ")
        number = input()
        if number == '1':
            test_row = Predict()
            test_row_scaled = scaler.transform(test_row)
            y_pred = knnclf.predict(test_row_scaled)
            if y_pred == 1:
                print("IRIS Setosa.")
            elif y_pred == 2:
                print("IRIS Versicolor.")
            else:
                print("IRIS Virginica.")
        elif number == '2':
            break
        else:
            print("Invalid input.")

def ClassifierModel():
    while True:
        print(" Enter the number of the classifier model you want to know its accuracy: ")
        print('\n 1- RANDOM FOREST CLASSIFIER')
        print('\n 2- KNeighborsClassifier')
        print("\n If you want to stop the program enter S ")
        task = input()
        if task == "1":
            MyRandomForestClassifier()
        elif task == "2":
            KNN()
        elif task.lower() == 's':
            break
        else:
            print("Invalid input. Please try again.")

VisualizationSpecies()
ClassifierModel()

#4.8	3.4	1.6	0.2	Iris-setosa
#5.6	2.5	3.9	1.1	Iris-versicolor
