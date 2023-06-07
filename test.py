import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler

is_part_A = False

if is_part_A:
    train_data = pd.read_csv("TrainingDataBinary.csv")
else:
    train_data = pd.read_csv("TrainingDataMulti.csv")

# Split the data into input features and output labels
X = train_data.iloc[:, :128]
y = train_data.iloc[:, 128]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data.iloc[:, :128])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


# Define the parameter search space
param_space = {
    'n_estimators': (1, 600),  # Number of trees in the forest
    'max_depth': (1, 50),  # Maximum depth of the trees
    'random_state': [42],  # Random seed for reproducibility
    'criterion': ['gini', 'entropy']
}

# Create the Random Forest Classifier
rf_classifier = RandomForestClassifier()


# Perform Bayesian optimization for parameter tuning
bayes_search = BayesSearchCV(
    rf_classifier, param_space, n_iter=75, cv=5, verbose=5, n_jobs=-1)
bayes_search.fit(X_train, y_train)

print("Best Parameters: ", bayes_search.best_params_)

rf_classifier.set_params(**bayes_search.best_params_)


# rf_classifier = RandomForestClassifier(
#    criterion='entropy', max_depth=13, n_estimators=79, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
# Print the best parameters and the corresponding accuracy score
# print("Best Parameters: ", bayes_search.best_params_)
# print("Best Accuracy: ", bayes_search.best_score_)

print('Model accuracy: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
