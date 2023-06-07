from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd


# Load the data and split it by columns to find the input and the output
data = pd.read_csv("TrainingDataBinary.csv", header=None)
X = data.iloc[:, :128]  # Inputs are columns 0-127
y = data.iloc[:, 128]  # Output is column 128

# Split the data into training data and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Defining svm
svm = SVC()

# Define the hyperparameter search space
param_space = {
    "C": (1e-4, 1e7, "log-uniform"),  # Range of values for C
    "gamma": (1e-9, 1e1, "log-uniform"),  # Range of values for gamma
}

# Bayesian optimisation with the param_space
# verbose is 5 to obtain relevant information
# n_jobs=-1 to utilize all available processors for parallel execution
# we are iterating 100 times an using a cross validation value of 5
opt = BayesSearchCV(
    svm, param_space, n_iter=100, cv=5, scoring="accuracy", verbose=5, n_jobs=-1
)

# Fitting the data
opt.fit(X_train, y_train)

# Calculating best params
best_params = opt.best_params_

# Creating a new SVC with the best parameters
svm2 = SVC(**best_params)

# Fitting the data
svm2.fit(X, y)

# Loading the test data
test_data = pd.read_csv("TestingDataBinary.csv", header=None)
X2 = test_data.iloc[:, :128]  # Inputs are columns 0-127

output = []

predictions = svm2.predict(X2)  # Predicting the outputs

# Combining the inputs to the output
for i in range(len(predictions)):
    output.append(list(X2.iloc[i]) + [predictions[i]])

# Defining the dataframe
df = pd.DataFrame(output)

# Creating the csv file and adding the results to it
df.to_csv('TestingResultsBinary.csv', header=False, index=False)
