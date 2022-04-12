## Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


## Reading the data set
train = pd.read_csv('/Users/gabrielvictorgomesferreira/Desktop/Diabetes_Project/Data/Final_Var_Eng_train.csv')


# Defining input and target variables
X = train.drop(['Diabetes_012'], axis = 1)
Y = train['Diabetes_012']

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)


# Defining top 8, 7, and 6 variables
# Train dataset Random Forest Classifier
RF_X_train_8 = X_train[['Interaction_1', 'Log_BMI', 'PhysHlth', 'MentHlth', 'Fruits', 'Age', 'Smoker', 'College_1_3']]
RF_X_train_7 = X_train[['Interaction_1', 'Log_BMI', 'PhysHlth', 'MentHlth', 'Fruits', 'Age', 'Smoker']]
RF_X_train_6 = X_train[['Interaction_1', 'Log_BMI', 'PhysHlth', 'MentHlth', 'Fruits', 'Age']]



## Defining the hyper-parameters for Random Forest Classifier
RF_param_grid = {'n_estimators': [100, 300, 500],
                 'min_samples_split': [10, 15], 
                 'min_samples_leaf': [5, 7], 
                 'max_depth' : [3, 5, 7]}

# Performing GridSearch
RF_grid_search_6 = GridSearchCV(RandomForestClassifier(), RF_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(RF_X_train_6, Y_train)
RF_grid_search_7 = GridSearchCV(RandomForestClassifier(), RF_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(RF_X_train_7, Y_train)
RF_grid_search_8 = GridSearchCV(RandomForestClassifier(), RF_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(RF_X_train_8, Y_train)


# Extracting the best model
RF_model_6 = RF_grid_search_6.best_estimator_
RF_score_6 = RF_grid_search_6.cv_results_
RF_score_6 = RF_score_6['mean_test_score'][0]

# Printing the results
print("Random Forest Classifier with the top 6 featues")
print("Model: ", RF_model_6)
print("Score: ", round(RF_score_6, 3))
print(" ")
print(" ")

# Extracting the best model
RF_model_7 = RF_grid_search_7.best_estimator_
RF_score_7 = RF_grid_search_7.cv_results_
RF_score_7 = RF_score_7['mean_test_score'][0]

# Printing the results
print("Random Forest Classifier with the top 7 featues")
print("Model: ", RF_model_7)      
print("Score: ", round(RF_score_7, 3))
print(" ")
print(" ")

# Extracting the best model
RF_model_8 = RF_grid_search_8.best_estimator_
RF_score_8 = RF_grid_search_8.cv_results_
RF_score_8 = RF_score_8['mean_test_score'][0]

# Printing the results
print("Random Forest Classifier with the top 8 featues")
print("Model: ", RF_model_8)
print("Score: ", round(RF_score_8, 3))
print(" ")
print(" ")

## Defining the hyper-parameters for Ada
# Train dataset AdaBoost with Decision Tree Classifier
Ada_X_train_8 = X_train[['Interaction_1', 'Log_BMI', 'Veggies', 'Stroke', 'Smoker', 'Age', 'PhysHlth', 'Interaction_5']]
Ada_X_train_7 = X_train[['Interaction_1', 'Log_BMI', 'Veggies', 'Stroke', 'Smoker', 'Age', 'PhysHlth']]
Ada_X_train_6 = X_train[['Interaction_1', 'Log_BMI', 'Veggies', 'Stroke', 'Smoker', 'Age']]


Ada_param_grid = {'n_estimators': [100, 300, 500],
                 'base_estimator__min_samples_split': [10, 15], 
                 'base_estimator__min_samples_leaf': [5, 7], 
                 'base_estimator__max_depth' : [3, 5, 7],
                 'learning_rate': [0.001, 0.01, 0.1]}

## Running grid search with 3 fold
Ada_grid_search_6 = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), Ada_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(Ada_X_train_6, Y_train)
Ada_grid_search_7 = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), Ada_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(Ada_X_train_7, Y_train)
Ada_grid_search_8 = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), Ada_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(Ada_X_train_8, Y_train)

# Extracting the best model
Ada_model_6 = Ada_grid_search_6.best_estimator_
Ada_score_6 = Ada_grid_search_6.cv_results_
Ada_score_6 = Ada_score_6['mean_test_score'][0]

# Printing the results
print("AdaBoost Classifier with the top 6 features")
print("Model: ", Ada_model_6)
print("Score: ", round(Ada_score_6, 3))
print(" ")
print(" ")

# Extracting the best model
Ada_model_7 = Ada_grid_search_7.best_estimator_
Ada_score_7 = Ada_grid_search_7.cv_results_
Ada_score_7 = Ada_score_7['mean_test_score'][0]

# Printing the results
print("AdaBoost Classifier with the top 7 features")
print("Model: ", Ada_model_7)
print("Score: ", round(Ada_score_7, 3))
print(" ")
print(" ")

# Extracting the best model
Ada_model_8 = Ada_grid_search_8.best_estimator_
Ada_score_8 = Ada_grid_search_8.cv_results_
Ada_score_8 = Ada_score_8['mean_test_score'][0]

# Printing the results
print("AdaBoost Classifier with the top 8 features")
print("Model: ", Ada_model_8)
print("Score: ", round(Ada_score_8, 3))
print(" ")
print(" ")

# Defining top 8, 7, and 6 variables
# Train dataset Decision Tree Classifier
Tree_X_train_8 = X_train[['Interaction_1', 'Log_BMI', 'Veggies', 'Smoker', 'PhysActivity', 'MentHlth', 'Interaction_5', 'HeartDiseaseorAttack']]
Tree_X_train_7 = X_train[['Interaction_1', 'Log_BMI', 'Veggies', 'Smoker', 'PhysActivity', 'MentHlth', 'Interaction_5']]
Tree_X_train_6 = X_train[['Interaction_1', 'Log_BMI', 'Veggies', 'Smoker', 'PhysActivity', 'MentHlth']]



## Defining the hyper-parameters for Decision Tree Classifier
tree_param_grid = {'min_samples_split': [10, 15], 
                 'min_samples_leaf': [5, 7], 
                 'max_depth' : [3, 5, 7]}

# Performing GridSearch
tree_grid_search_6 = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(Tree_X_train_6, Y_train)
tree_grid_search_7 = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(Tree_X_train_7, Y_train)
tree_grid_search_8 = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, cv = 3, scoring = 'f1_micro', n_jobs = -1).fit(Tree_X_train_8, Y_train)

# Extracting the best model
Tree_model_6 = tree_grid_search_6.best_estimator_
Tree_score_6 = tree_grid_search_6.cv_results_
Tree_score_6 = Tree_score_6['mean_test_score'][0]

# Printing the results
print("Decision Tree Classifier with the top 6 features")
print("Model: ", Tree_model_6)
print("Score: ", round(Tree_score_6, 3))
print(" ")
print(" ")

# Extracting the best model
Tree_model_7 = tree_grid_search_7.best_estimator_
Tree_score_7 = tree_grid_search_7.cv_results_
Tree_score_7 = Tree_score_7['mean_test_score'][0]

# Printing the results
print("Decision Tree Classifier with the top 7 features")
print("Model: ", Tree_model_7)
print("Score: ", round(Tree_score_7, 3))
print(" ")
print(" ")

# Extracting the best model
Tree_model_8 = tree_grid_search_8.best_estimator_
Tree_score_8 = tree_grid_search_8.cv_results_
Tree_score_8 = Tree_score_8['mean_test_score'][0]

# Printing the results
print("Decision Tree Classifier with the top 8 features")
print("Model: ", Tree_model_8)
print("Score: ", round(Tree_score_8, 3))
print(" ")
print(" ")