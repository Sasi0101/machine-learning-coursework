import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Loading the training data
training_data = pd.read_csv("TrainingDataBinary.csv", header=None)

# Spliting the data into input and output
X = training_data.iloc[:, :128]  # Inputs are columns 0-127
y = training_data.iloc[:, 128]  # Output is column 128

# Initializing the random forest classifier
random_forest_classifier = RandomForestClassifier(
    criterion='entropy', max_depth=47, n_estimators=136, random_state=42)

# Fittig the model
random_forest_classifier.fit(X, y)

# Loading the testing data
testing_data = pd.read_csv("TestingDataBinary.csv", header=None)
X2 = testing_data.iloc[:, :128]  # Inputs are columns 0-127

# Making a prediction
predictions = random_forest_classifier.predict(X2)
print("Predicitons: ", predictions)
output = []  # All the output will be stored here

# Combining the inputs to the output
for i in range(len(predictions)):
    output.append(list(X2.iloc[i]) + [predictions[i]])

# Defining the dataframe
data_frame = pd.DataFrame(output)

# Creating the csv file and adding the results to it
data_frame.to_csv('TestingResultsBinary.csv', header=False, index=False)
