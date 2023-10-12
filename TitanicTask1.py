#Titanic Task 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pandas import *
import warnings
warnings.filterwarnings('ignore')

#Reading CSV FILE
data = read_csv("tested.csv")
print(data.head())

#Pre-processing for the data
print(data.info())

#We concluded from the information that there is 3 columns have nulls: Age, Fare and Cabin ..
#so we will drop all nulls, as they are not substantial, we dropped 2 columns Cabin and Age as they were having a lot of null rows
#and we dropped one row in the whole data because it was null in fare
data = data.drop(['Cabin', 'Age'], axis=1)
data = data.dropna()
#Encode the gender
Sex = data['Sex'].replace(['female', 'male'], [1, 0], inplace=True)
#'Name', 'Ticket' and 'Embarked' are categorical columns, we will need to encode them or drop them.
data = data.drop(['Name', 'Ticket', 'Embarked'], axis=1)
print(data.info())

#We now have 417 rows X 7 columns with clear data
#We will calculate the percentage of passengers who were survived and not.
total_count = len(data['Survived'])
survived_count = (data['Survived'] == 1).sum()
percentage_survived = (survived_count / total_count) * 100
percentage_dead = 100 - percentage_survived
print(f"Percentage of people who survived: {percentage_survived:.2f}%")
print(f"Percentage of people who dead: {percentage_dead:.2f}%")

# Train and test split --> stratified
X = data.drop('Survived', axis=1).copy()
y = data['Survived'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

#We will use more than one model to settle in the end with the one which got the higher accuracy.
#Firstly, We will do function which calculates the accuracy of each model to be more easier.
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

def Predict():
    # Getting input from the user to know if he/ she survived or not.. It's optional.
    print('\n Enter the passenger ID you want to know survived or not:')
    pid = int(input())
    print("\n Enter its PClass: ")
    pclass = int(input())
    print("\n Enter its sex: ")
    sex = int(input())
    print("\n Enter the number of its siblings: ")
    sib = int(input())
    print("\n Enter its parch: ")
    parch = int(input())
    print("\n Enter its fare: ")
    fare = float(input())
    test_row = [[pid, pclass, sex, sib, parch, fare]]
    return test_row

# 1- RANDOM FOREST CLASSIFIER Model
def MyRandomForestClassifier():
    rf = RandomForestClassifier(max_depth=6, n_estimators=100)
    rclf = rf.fit(X_train,y_train)
    y_pred1 = rclf.predict(X_test)
    print('\n 1- RANDOM FOREST CLASSIFIER')
    print('\n-------------------- Training accuracy --------------------')
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y_train, rclf.predict(X_train))))
    CalculateAccuracy(y_test, y_pred1)
    while True:
        print("Do you want to know if specific passenger is survived press 1, if not press 2: ")
        number = input()
        if number == '1':
            test_row = Predict()
            y_pred = rclf.predict(test_row)
            if y_pred == 1:
                print("Survived.")
            else:
                print("Dead.")
        elif number == '2':
            break
        else:
            print("Invalid input.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
        print("Do you want to know if specific passenger is survived press 1, if not press 2: ")
        number = input()
        if number == '1':
            test_row = Predict()
            test_row_scaled = scaler.transform(test_row)
            y_pred = knnclf.predict(test_row_scaled)
            if y_pred == 1:
                print("Survived.")
            else:
                print("Dead.")
        elif number == '2':
            break
        else:
            print("Invalid input.")

# 3- XGBoost Model
def XGBoost():
    xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, missing=1, early_stopping_rounds=10,
    learning_rate=0.1, max_depth=3, gamma=5, eval_metric=['merror', 'mlogloss'], seed=42)

    xgb_clf.fit(X_train, y_train, verbose=0, eval_set=[(X_train, y_train), (X_test, y_test)])
    y_pred3 = xgb_clf.predict(X_test)
    print('\n 3- The XGBoost Model')
    print('\n-------------------- Training accuracy --------------------')
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y_train, xgb_clf.predict(X_train))))
    CalculateAccuracy(y_test, y_pred3)
    while True:
        print("Do you want to know if specific passenger is survived press 1, if not press 2: ")
        number = input()
        if number == '1':
            test_row = Predict()
            y_pred = xgb_clf.predict(test_row)
            if y_pred == 1:
                print("Survived.")
            else:
                print("Dead.")
        elif number == '2':
            break
        else:
            print("Invalid input.")
def ClassifierModel():
    while True:
        print(" Enter the number of the classifier model you want to know its accuracy: ")
        print('\n 1- RANDOM FOREST CLASSIFIER')
        print('\n 2- KNeighborsClassifier')
        print('\n 3- The XGBoost Model')
        print("\n If you want to stop the program enter S ")
        task = input()
        if task == "1":
            MyRandomForestClassifier()
        elif task == "2":
            KNN()
        elif task == "3":
            XGBoost()
        elif task.lower() == 's':
            break
        else:
            print("Invalid input. Please try again.")

ClassifierModel()

#893	1	3	Wilkes, Mrs. James (Ellen Needs)	female	47	1	0	363272	7		S
#909	0	3	Assaf, Mr. Gerios	male	21	0	0	2692	7.225		C

