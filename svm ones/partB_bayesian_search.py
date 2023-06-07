from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd


# Load the data and split it by columns to find the input and the output
data = pd.read_csv("TrainingDataMulti.csv", header=None)
X = data.iloc[:, :128]  # Inputs are columns 0-127
y = data.iloc[:, 128]  # Output is column 128

# Split the data into training data and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Defining svm
svm = SVC()

# Define the hyperparameter search space
param_space = {
    "C": (1e-4, 1e8, "log-uniform"),  # Range of values for C
    "gamma": (1e-8, 1e2, "log-uniform"),  # Range of values for gamma
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

# Calculating best params and best scores
best_params = opt.best_params_
best_score = opt.best_score_

# Finding the best model and the score for that
best_model = opt.best_estimator_
validation_accuracy = best_model.score(X_val, y_val)
training_accuracy = best_model.score(X_train, y_train)
training_error = 1 - training_accuracy

# Printing out the values
print("Best Hyperparameters: ", best_params)
print("Best Score: ", best_score)
print("Validation Accuracy: ", validation_accuracy)
print("Training accuracy: ", training_accuracy)
print("Training error: ", training_error)

'''
print("====================== Starting svm2 ======================")

# Creating a new SVC with the best parameters
svm2 = SVC(**best_params)

# Fitting the data
svm2.fit(X, y)

# Test data
test_data = pd.read_csv("testzeros.csv", header=None)
X2 = test_data.iloc[:, :128]
y2 = test_data.iloc[:, 128]

# Predicting the data
predictions = svm2.predict(X2)


# Check to see how many are correct
correct_predictions = 0

for i in range(len(predictions)):
    if y2[i] == predictions[i]:
        correct_predictions += 1

print("Number of correct predictions are: ",
      correct_predictions, "/", len(predictions))
'''
