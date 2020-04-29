# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 08:30:53 2020

@author: Soumen
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier
from sklearn import linear_model

# Print out the first 5 lines from the transfusion.data file
#!head -n5 datasets/transfusion.data

# Read in dataset
transfusion = pd.read_csv("transfusion.data")

# Print out the first rows of our dataset
# ... YOUR CODE FOR TASK 2 ...
transfusion.head()

# Print a concise summary of transfusion DataFrame
# ... YOUR CODE FOR TASK 3 ...
transfusion.info()

# Rename target column as 'target' for brevity 
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

# Print out the first 2 rows
# ... YOUR CODE FOR TASK 4 ...
transfusion.head(2)

# Print target incidence proportions, rounding output to 3 decimal places
# ... YOUR CODE FOR TASK 5 ...
transfusion.target.value_counts(normalize=True).round(3)

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)

# Print out the first 2 rows of X_train
# ... YOUR CODE FOR TASK 6 ...
X_train.head(2)

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')
    
# X_train's variance, rounding the output to 3 decimal places
# ... YOUR CODE FOR TASK 8 ...
X_train.var().round(3)

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed= X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = 'Monetary (c.c. blood)'

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns= col_to_normalize, inplace=True)

# Check the variance for X_train_normed
# ... YOUR CODE FOR TASK 9 ...
X_train_normed.var().round(3)


# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the model
logreg.fit(X_train_normed, y_train)

# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')

# Importing itemgetter

# Sort models based on their AUC score from highest to lowest
sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    reverse= True
)