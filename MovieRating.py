from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#Reading CSV FILE
data = read_csv("Movies.csv", encoding='latin-1')
print(data.head())
print(data.info())

#we concluded that we have missing values in all columns except name, so we will try to handle the missing data

# Calculate the mode of duration column
mode_value = data['Duration'].mode()[0]
# Impute missing values in duration column with its mode
data['Duration'].fillna(mode_value, inplace=True)

#We will drop all the nulls.
data = data.dropna()

# Check for remaining missing values
print(data.info())

# Custom function to extract numeric year
def extract_numeric_year(year_str):
    year_str = str(year_str)  # Ensure it's a string
    match = re.search(r'\d{4}', year_str)
    if match:
        return int(match.group(0))
    return None

# Apply the custom function to clean and extract numeric years
data['Year'] = data['Year'].apply(extract_numeric_year)
# print('year', data['Year'])

# Split the 'Genres' column by ','
list_Genre = []
list_Genre = data['Genre'].str.split(",")

Genre=[]
for x in list_Genre:
  Genre.extend(x)

#print('Genre', Genre)
Genres = [i.strip() for i in Genre]
# print('Genres', Genres)
data2 = pd.DataFrame(Genres)
# print(data2)
data2.columns = ["Genre"]

# Use value_counts to count the occurrences of each genre
genre_counts = data2['Genre'].value_counts()

# Print the genre counts
print('genre counts', genre_counts)
listOfGenre = data2["Genre"].unique()
plt.pie(genre_counts, labels=listOfGenre)
plt.title('Count of genre')
plt.show()

# Group the data by 'Director' and calculate the mean rating
director_avg_rating = data.groupby('Director')['Rating'].mean()

# Print the average ratings for each director
print(director_avg_rating)

# Group the data by 'Actor 1' and calculate the mean rating
actor1_avg_rating = data.groupby('Actor 1')['Rating'].mean()

# Group the data by 'Actor 2' and calculate the mean rating
actor2_avg_rating = data.groupby('Actor 2')['Rating'].mean()

# Group the data by 'Actor 3' and calculate the mean rating
actor3_avg_rating = data.groupby('Actor 3')['Rating'].mean()

# Print the average ratings for each actor
print(actor1_avg_rating)
print(actor2_avg_rating)
print(actor3_avg_rating)

# Group the data by release year and calculate the average rating for each year
yearly_avg_rating = data.groupby('Year')['Rating'].mean()

# Create a line plot to visualize the Average Movie Rating Over the Years
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg_rating.index, yearly_avg_rating.values, marker='o', linestyle='-')
plt.title('Average rating of movies over the years')
plt.xlabel('Release Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

listOfRate = []
listOfRate = data['Rating'].values.tolist()
print('list of rate', listOfRate)

avg_ratings_movies = data.groupby('Genre')['Rating'].mean()
print(avg_ratings_movies)

plt.figure(figsize=(10, 6))
plt.plot(avg_ratings_movies.index, avg_ratings_movies.values, marker='o', linestyle='-')
plt.title('Average rating of all movies')
plt.xlabel('ratings')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

label_encoders = {}
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    data[feature] = label_encoders[feature].fit_transform(data[feature])

X = data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def Predict():
    # Getting input from the user to know the specific rating of the movie.. It's optional.
    print('\n Enter Genre of the movie:')
    genre = input()
    print("\n Enter the director's name: ")
    director = input()
    print("\n Enter actor one's name: ")
    actor1 = input()
    print("\n Enter actor two's name: ")
    actor2 = input()
    print("\n Enter actor three's name: ")
    actor3 = input()
    # Create a dictionary with feature names and values
    input_data = {
        'Genre': [genre],
        'Director': [director],
        'Actor1': [actor1],
        'Actor2': [actor2],
        'Actor3': [actor3]
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Perform one-hot encoding for categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3'])

    # Make sure the order of columns matches the model's input
    input_df_encoded = input_df_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Convert the DataFrame to a numpy array
    test_row = input_df_encoded.values
    return test_row
# 1- Linear Regression Model
def linearModel():
    model = LinearRegression()
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Calculate the Root Mean Squared Error (RMSE) on the training data
    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    r_squared = r2_score(y_test, y_pred)
    print('\n 1- LINEAR REGRESSION MODEL')
    print('RMSE on Training Data: {:.2f}'.format(rmse_train))
    print('\n-------------------- Testing RMSE --------------------')
    print('RMSE on Testing Data: {:.2f}'.format(rmse))
    print(f'Mean Squared Error: {mse}')
    print(f"R-squared: {r_squared}")
    while True:
        print("Do you want to know the rating of specific movie press 1, if not press 2: ")
        number = input()
        if number == '1':
            test_row = Predict()
            y_pred = model.predict(test_row)
            print("Predicted rating: ", y_pred)
        elif number == '2':
            break
        else:
            print("Invalid input.")
    return y_pred

# 2- RANDOM FOREST CLASSIFIER Model
def MyRandomForestRegressor():
    rf = RandomForestRegressor(max_depth=6, n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('\n 2- RANDOM FOREST REGRESSOR')
    # Calculate and display regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)
    print('\n-------------------- Training RMSE --------------------')
    print('RMSE on Training Data: {:.2f}'.format(np.sqrt(mean_squared_error(y_train, rf.predict(X_train)))))

    print('\n-------------------- Testing RMSE --------------------')
    print('RMSE on Testing Data: {:.2f}'.format(rmse))
    print(f'Mean Squared Error: {mse}')
    print(f'r squared', {r_squared})
    while True:
            print("Do you want to know the rating of specific movie press 1, if not press 2: ")
            number = input()
            if number == '1':
                test_row = Predict()
                y_pred = rf.predict(test_row)
                print("Predicted rating: ", y_pred)
            elif number == '2':
                break
            else:
                print("Invalid input.")


def ClassifierModel():
    while True:
        print(" Enter the number of the classifier model you want to know its RMSE: ")
        print('\n 1- LINEAR REGRESSION MODEL')
        print('\n 2- RANDOM FOREST REGRESSOR')
        print("\n If you want to stop the program enter S ")
        task = input()
        if task == "1":
            linearModel()
        elif task == "2":
            MyRandomForestRegressor()
        elif task.lower() == 's':
            break
        else:
            print("Invalid input. Please try again.")


ClassifierModel()
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred)
plt.title("Actual vs. predicted movie ratings with regression line")
plt.xlabel("Actual Ratings (y_test)")
plt.ylabel("Predicted Ratings (y_pred)")
plt.show()
#the features to predict with Drama, Gaurav, Bakshi, Rasika Dugal, Vivek Ghamand, Arvind Jangid
#96 min	Drama	6.2	17	Madhu Ambat	Rati Agnihotri	Gulshan Grover	Atul Kulkarni
#100% Love	-2012	166 min	Comedy, Drama, Romance	5.7	512	Rabi Kinagi	Jeet	Koyel Mallick	Sujoy Ghosh

