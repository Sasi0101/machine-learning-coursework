from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

# Loading the training data
data = pd.read_csv("TrainingDataMulti.csv", header=None)
X = data.iloc[:, :128]  # Inputs are columns 0-127
y = data.iloc[:, 128]  # Output is column 128

# Creating the SVC
svm = SVC(C=31086013.464533035, gamma=1.5561510817757932e-06)

# Fitting the SVC
svm.fit(X, y)

# Loading the test data
test_data = pd.read_csv("TestingDataMulti.csv", header=None)
X2 = test_data.iloc[:, :128]  # Inputs are columns 0-127


predictions = svm.predict(X2)

# Check to see how many are correct
correct_predictions = 0

# for i in range(len(predictions)):
#    if y2[i] == predictions[i]:
#        correct_predictions += 1

print("Predictions: ", predictions)

print("Number of correct predictions are: ",
      correct_predictions, "/", len(predictions))


'''
output = []

predictions = svm.predict(X2)  # Predicting the outputs

# Combining the inputs to the output
for i in range(len(predictions)):
    output.append(list(X2.iloc[i]) + [predictions[i]])

# Defining the dataframe
df = pd.DataFrame(output)

# Creating the csv file and adding the results to it
df.to_csv('TestingResultsMulti.csv', header=False, index=False)
'''
