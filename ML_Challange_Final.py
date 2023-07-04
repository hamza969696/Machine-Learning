# %%
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# %%
# Read csv file into dataframe
data = pd.read_csv("Newdata-2.csv")

# Fix labels
data = data.rename(columns = {'Ad_Campaign_1' : 'Ad_Campaign1'})

# Fix types: object to categorical
data.Customer_Type = pd.Categorical(data.Customer_Type)

# %%
# Are there missing values?
print(data.isnull().sum())

# Are there any duplicated rows?
duplicates = data.duplicated()
print(duplicates.any())

# %%
#### --------- Encode categorical variables (Customer_Type) --------- ####
# Create instance of one-hot encoder
encoder = OneHotEncoder(categories='auto', drop='first')

# Fit encoder to the column
encoded_features = encoder.fit_transform(data[['Customer_Type']])

# Convert encoded features to a dataframe and concatenate it to the original dataframe
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(['Customer_Type']))
data = pd.concat([data, encoded_df], axis=1)
data = data.drop('Customer_Type', axis=1)

# %%
#### --------- Split data (training, validation, test) ---------  ####
# Reserve 20% for testing
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Split into train (60%) and validation (20%)
train, validation = train_test_split(train, test_size=0.25, random_state=42)

# Print the sizes of the resulting sets
print("Training set size:", len(train))
print("Validation set size:", len(validation))
print("Test set size:", len(test))

# %%
#### --------- Detect outliers with boxplot method on TRAINING data --------- ####
fig, ax = plt.subplots(figsize=(8, 6))

# Add whiskers, mean, and outliers
boxplot = train.boxplot(ax=ax, sym='k+', showmeans=True, meanline=True, showfliers=True)

# Separate the variable labels
ax.set_xticks(range(1, len(train.columns) + 1))
ax.set_xticklabels(train.columns, rotation=45, ha='right')

# Add labels and title
ax.set_xlabel("Variables")
ax.set_ylabel("Values")
ax.set_title("Boxplot of Variables")

# Adjust the spacing between subplots if needed
plt.tight_layout()

# Show the modified boxplot
plt.show()

# %%
# Remove outlier from TRAINING set 
index_outlier = train[train['ProductPage_Time'] == 63973.52223].index[0]

train['ProductPage_Time'] = train['ProductPage_Time'].drop(index_outlier)

# %%
#### --------- Detect outliers with boxplot method on VALIDATION data --------- ####
fig, ax = plt.subplots(figsize=(8, 6))

# Add whiskers, mean, and outliers
boxplot = validation.boxplot(ax=ax, sym='k+', showmeans=True, meanline=True, showfliers=True)

# Separate the variable labels
ax.set_xticks(range(1, len(validation.columns) + 1))
ax.set_xticklabels(validation.columns, rotation=45, ha='right')

# Add labels and title
ax.set_xlabel("Variables")
ax.set_ylabel("Values")
ax.set_title("Boxplot of Variables")

# Adjust the spacing between subplots if needed
plt.tight_layout()

# Show the modified boxplot
plt.show()

# %%
#### --------- Detect outliers with boxplot method on TEST data --------- ####
fig, ax = plt.subplots(figsize=(8, 6))

# Add whiskers, mean, and outliers
boxplot = test.boxplot(ax=ax, sym='k+', showmeans=True, meanline=True, showfliers=True)

# Separate the variable labels
ax.set_xticks(range(1, len(test.columns) + 1))
ax.set_xticklabels(test.columns, rotation=45, ha='right')

# Add labels and title
ax.set_xlabel("Variables")
ax.set_ylabel("Values")
ax.set_title("Boxplot of Variables")

# Adjust the spacing between subplots if needed
plt.tight_layout()

# Show the modified boxplot
plt.show()

# %%
#### --------- Scaling: Normalization --------- ####
# Normalize TRAINING, VALIDATION and TEST sets
# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler on the training set and obtain the scaling parameters
scaler.fit(train)

# Normalize the training set using the computed scaling parameters
train_norm = pd.DataFrame(scaler.transform(train), columns=train.columns)
train = train_norm

# Normalize the validation set using the same scaling parameters
validation_norm = pd.DataFrame(scaler.transform(validation), columns=validation.columns)
validation = validation_norm

# Normalize the test set using the same scaling parameters
test_norm = pd.DataFrame(scaler.transform(test), columns=test.columns)
test = test_norm

# %%
#### --------- Class imbalance: Oversampling --------- ####
# Check whether the target variable 'Transaction' is imbalanced
print(train.Transaction.value_counts())

# Replace missing value with mean of the column (after having removed an outlier in the training set)
train = train.fillna(train.mean())
train.isna().sum()

# Separate feastures and target variable
X = train.drop('Transaction', axis=1)
y = train['Transaction']

# Create instance of SMOTE oversampler
smote = SMOTE()

# Apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine into new dataframe
resampled_train = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Transaction')], axis=1)
train = resampled_train

# %%
#### --------- Feature selection --------- ####
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled = pd.Series(y_resampled, name='Transaction')

# Feature selection
model = LogisticRegression()

model.fit(X_resampled,y_resampled)

# Compute feature importances using permutation importance
results = permutation_importance(model, X_resampled, y_resampled, scoring='accuracy')

importance_scores = pd.DataFrame({'Feature': X_resampled.columns,
                                   'Importance': results.importances_mean})
importance_scores = importance_scores[importance_scores['Importance'] > 0]
print(importance_scores)
#consider only those features which have an importance score of more than 0

# %%
# df after feature selection
selected_features = [0, 1, 2, 3, 4, 5, 6, 8, 9,10, 11, 16, 21]
X_train_selected = X_resampled.iloc[:, selected_features]
y_val=validation['Transaction']
X_val=validation.drop('Transaction',axis=1)
X_val_selected = X_val.iloc[:, selected_features]
y_test=test['Transaction']
X_test=test.drop('Transaction',axis=1)
X_test_selected = X_test.iloc[:, selected_features]

# %%
#### --------- Logistic Regreggion --------- ####
#logistic model with relevant features
#train the data and hyperparameter tuning using gridsearchcv
parameters = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
model = LogisticRegression()
model.fit(X_train_selected, y_resampled)

grid_search = GridSearchCV(model, parameters, cv=3)

# fit the grid search object on the training data
grid_search.fit(X_train_selected, y_resampled)

print(grid_search.best_params_)

# Selected features: {'C': 1, 'penalty': 'l2'}

#%%
# Check accuracy on training set
model2 = LogisticRegression(C=1,penalty='l2')
model2.fit(X_train_selected, y_resampled)
y_train_predict=model2.predict(X_train_selected)
accuracy_logistic_training = accuracy_score(y_resampled, y_train_predict)
print('Accuracy score - logistic regression - training set: ')
print(accuracy_logistic_training)

# Check accuracy on validation set
y_val_predict = model2.predict(X_val_selected)
accuracy_logistic_val = accuracy_score(y_val, y_val_predict)
print('Accuracy score - logistic regression - validation set: ')
print(accuracy_logistic_val)

# Confusion matrix
c_m_logistic = confusion_matrix(y_val, y_val_predict)
print('Confusion matrix: ')
print(c_m_logistic)

# Precision, recall and f1 score
precision = precision_score(y_val, y_val_predict)
recall = recall_score(y_val, y_val_predict)
f1 = f1_score(y_val, y_val_predict)

print('Precision: ')
print(precision)
print('Recall: ')
print(recall)
print('F1 score: ')
print(f1)

# Check accuracy on test set
y_test_predict=model2.predict(X_test_selected)
accuracy_logistic_test = accuracy_score(y_test, y_test_predict)
print('Accuracy score - logistic regression - test set: ')
print(accuracy_logistic_test)

# %%
# Create a heatmap of the confusion matrix
sns.heatmap(c_m_logistic, annot=True, fmt='d', cmap='Blues')

# Customize the plot
plt.title('Confusion Matrix Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show the plot
plt.savefig('confusion_matrix_logistic.png')
plt.show()

# %%
#### --------- Support Vector Machine --------- ####
##SVM model
svm_model = SVC()

# Parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

# GridSearchCV object
grid_search = GridSearchCV(svm_model, param_grid, cv=3)

# Fit the grid search object on the training data
grid_search.fit(X_train_selected, y_resampled)

# %% 
# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Create a new SVM model with the best parameters
svm_model_best = SVC(**best_params)

# Train the SVM model with the best parameters on the selected features
svm_model_best.fit(X_train_selected, y_resampled)

# %%
# Make predictions on training set 
y_train_predict_svm = svm_model_best.predict(X_train_selected)

# Calculate accuracy on the training set 
accuracy_train = accuracy_score(y_resampled, y_train_predict_svm)
print("Training Accuracy:", accuracy_train)

# Make predictions on the validation set
y_val_predict_svm = svm_model_best.predict(X_val_selected)

# Calculate the accuracy on the validation set
accuracy_val = accuracy_score(y_val, y_val_predict_svm)
print("Validation Accuracy:", accuracy_val)

# Make predictions on the test set
y_test_predict_svm = svm_model_best.predict(X_test_selected)

# Calculate the accuracy on the test set
accuracy_test = accuracy_score(y_test, y_test_predict_svm)
print("Test Accuracy:", accuracy_test)

# %%
# Confusion matrix
c_m_svm = confusion_matrix(y_val, y_val_predict_svm)
print('Confusion matrix SVM: ')
print(c_m_svm)

f1_SVM = f1_score(y_val, y_val_predict_svm)
print(f1_SVM)
precision_svm = precision_score(y_val, y_val_predict_svm)
recall_svm = recall_score(y_val, y_val_predict_svm)
f1_svm = f1_score(y_val, y_val_predict_svm)

print('Precision SVM: ')
print(precision_svm)
print('Recall SVM: ')
print(recall_svm)
print('F1 score SVM: ')
print(f1_svm)

# %%
# Create a heatmap of the confusion matrix
sns.heatmap(c_m_svm, annot=True, fmt='d', cmap='Blues')

# Customize the plot
plt.title('Confusion Matrix SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show the plot
plt.savefig('confusion_matrix_SVM.png')
plt.show()

# %% 
# #### --------- Decision Tree --------- ####

# Plotting the decision tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_selected, y_resampled)

# %%
# Hyperparameter tuning with gridsearch, choice between gini impurity/entropy and maximum depth of the tree
parameters_dt_model = {'criterion': ['gini','entropy'], 'max_depth': [None, 5, 10, 15]}
grid_search = GridSearchCV(dt_model, parameters_dt_model, cv=3)
grid_search.fit(X_train_selected, y_resampled)
print(grid_search.best_params_)

# %%
# Make predictions on the training set
y_train_predict_dt_model = dt_model.predict(X_train_selected)

# %%
# Calculate accuracy on the training set
accuracy_train_dt_model = accuracy_score(y_resampled, y_train_predict_dt_model)
print("Accuracy score - decision tree - training set:", accuracy_train_dt_model)

# %%
# Calculate accuracy on the validation set
y_val_predict_dt_model = dt_model.predict(X_val_selected)
accuracy_val_dt_model = accuracy_score(y_val, y_val_predict_dt_model)
print('Accuracy score - decision tree - validation set: ', accuracy_val_dt_model)

# %%
# Make predictions on the test set and calculate accuacy
y_test_predict_dt_model = dt_model.predict(X_test_selected)
accuracy_test_dt_model = accuracy_score(y_test, y_test_predict_dt_model)
print("Accuracy score - decision tree - test set:", accuracy_test_dt_model)

# %%
# Confusion matrix for decision tree
cm_dt_model = confusion_matrix(y_val, y_val_predict_dt_model)
print('Confusion matrix: ')
print(cm_dt_model)

# %%
# Create a heatmap of the confusion matrix
sns.heatmap(cm_dt_model, annot=True, fmt='d', cmap='Blues')

# Customize the plot
plt.title('Confusion Matrix Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show the plot
plt.savefig('confusion_matrix_DT.png')
plt.show()

# %%
# F1 score for decision tree
f1_dt = f1_score(y_val, y_val_predict_dt_model)
print(f1_dt)

# %%
precision_dt = precision_score(y_val, y_val_predict_dt_model)
recall_dt = recall_score(y_val, y_val_predict_dt_model)

print('Confusion matrix Decision Tree: ')
print(cm_dt_model)
print('F1 score Decision Tree: ')
print(f1_dt)
print('Precision Decision Tree: ')
print(precision_dt)
print('Recall Decision Tree: ')
print(recall_dt)

# %%
## Create CSV file with predictions
df = pd.DataFrame({'True Labels': y_test, 'Predicted Labels': y_test_predict_svm})

df.to_csv('Predictions.csv', index=False)
