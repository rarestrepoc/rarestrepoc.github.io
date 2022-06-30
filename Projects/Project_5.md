# Understanding and Predicting Property Maintenance Fines

In this project, we are going to design a predictive model using a dataset from the Michigan Data Science Team (MDST) that have data from blight violations are issued by the city to individuals who allow their properties to remain in a deteriorated condition.

We want to understanding when and why a resident might fail to comply with a blight ticket, this is where predictive modeling comes in. 

The datasets are: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.

### Data fields

train.csv & test.csv

- ticket_id - unique identifier for tickets
- agency_name - Agency that issued the ticket
- inspector_name - Name of inspector that issued the ticket
- violator_name - Name of the person/organization that the ticket was issued to
- violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
- mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
- ticket_issued_date - Date and time the ticket was issued
- hearing_date - Date and time the violator's hearing was scheduled
- violation_code, violation_description - Type of violation
- disposition - Judgment and judgement type
- fine_amount - Violation fine amount, excluding fees
- admin_fee - 20 USD fee assigned to responsible judgments
- state_fee - 10 USD fee assigned to responsible judgments
- late_fee - 10% fee assigned to responsible judgments
- discount_amount - discount applied, if any
- clean_up_cost - DPW clean-up or graffiti removal cost
- judgment_amount - Sum of all fines and fees
- grafitti_status - Flag for graffiti violations

train.csv only

- payment_amount - Amount paid, if any
- payment_date - Date payment was made, if it was received
- payment_status - Current payment status as of Feb 1 2017
- balance_due - Fines and fees still owed
- collection_status - Flag for payments in collections
- compliance [target variable for prediction] 
 - Null = Not responsible
 - 0 = Responsible, non-compliant
 - 1 = Responsible, compliant
- compliance_detail - More information on why each ticket was marked compliant or non-compliant




```python
'''
Libraires
'''
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2

'''
Cleaning functions
'''

def clean_1(text):
    pattern = '([\w ,]* [A-Z]+) *\d*'
    cltext = re.findall(pattern, text)
    return cltext[0]

def set_type_and_fill(X):
    X = X.fillna('missing')
    X = X.astype('str')
    return X 

'''
Training data
'''

df_train = pd.read_csv('datasets/train.csv', 
                       encoding='cp1252', 
                       low_memory=False, 
                       na_values='')

df_train = df_train[df_train['compliance'].eq(1) | df_train['compliance'].eq(0)]


'''
Test data
'''

data_test = pd.read_csv('datasets/test.csv').drop(
    ['ticket_id', 
     'violation_zip_code',
     'grafitti_status',
     'mailing_address_str_number',
     'non_us_str_code'], 
    axis=1
)

'''
Drop unnecessary and problematic (by format) columns
'''

X = df_train.drop(
    ['compliance', 
     'ticket_id', 
     'payment_amount', 
     'payment_date', 
     'payment_status', 
     'balance_due',
     'collection_status',
     'compliance_detail',
     'grafitti_status',
     'violation_zip_code',
     'mailing_address_str_number',
     'non_us_str_code'], 
    axis=1
)

y = df_train['compliance']

'''
Split data. 

Compliance:

0 --> positive class: does not pay the fine
1 --> negative class: pay the fine
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

'''
We are going to take only a small sample (n=500) to train several models and 
choose one. We do this because of the size of the training set.
'''
_ = pd.merge(X_train, y_train, left_index=True, right_index=True).sample(n=500)
X_sample = _.iloc[:,:-1]
y_sample = _.iloc[:,-1]
```

We are going to create a pipeline to pre-process the data so that the predictive model works correctly.


```python
'''
First we need to separate the categorical from the numeric columns.
'''

categorical = list(X.select_dtypes('O').columns)
numerical = list(X.select_dtypes('number').columns)

'''
Designing the preprocessor
'''

preprocessor = ColumnTransformer(
    # preprocess all categorial columns
    [
     ('cat',
      # Pipeline to process categorical features
      Pipeline(
         [
             ('set_type_and_fill', FunctionTransformer(set_type_and_fill)),
             ('categorical', OneHotEncoder(handle_unknown='ignore'))
          
         ]
      ), 
      categorical),    
     ('num',
      # Pipeline to process numerical features
      Pipeline(
         [
          ('imputer', SimpleImputer(strategy='constant', fill_value=0))
         ]
      ), 
      numerical)
    ] 
)
```

Now we are going to train 4 different classifying models to have a point of comparison and we are going to choose the one that gives us the largest AUC. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.

The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.

For each classifier we will use GridSearch to find the best model parameters and the maximum possible AUC.


```python
'''
Kernelized Support Vector Machine with RBF kernel
'''

svc_pipe = Pipeline(
    [
     ('processor', preprocessor),  # Preprocess data
     ('best', SelectKBest(score_func=chi2)), # Select k best characteristics
     ('svc', SVC(kernel='rbf', random_state=0)) # Classification model
    ]
)

'''
Parametters for GridSearch
'''

parametters_svc = {
    'best__k':[10, 50, 100],
    'svc__C':[0.001, 0.1, 1, 10, 100],
    'svc__gamma':[0.001, 0.01, 0.1, 1, 10, 100]
}

grid_svc_auc = GridSearchCV(svc_pipe, 
                            param_grid=parametters_svc, 
                            scoring='roc_auc', 
                            n_jobs=-1)

grid_svc_auc.fit(X_sample, y_sample)


'''
Random Forest Classifier
'''

rfc_pipe = Pipeline(
    [
     ('processor', preprocessor),
     ('best', SelectKBest(score_func=chi2)),
     ('rfc', RandomForestClassifier(n_jobs=-1, random_state=0))
    ]
)

parametters_rfc = {
    'best__k':[10, 50, 100],
    'rfc__n_estimators':[50, 100, 150], 
    'rfc__max_features':[1, 2], 
    'rfc__max_depth':[2, 3, 4, 5]
}

grid_rfc_auc = GridSearchCV(rfc_pipe, 
                            param_grid=parametters_rfc, 
                            scoring='roc_auc', 
                            n_jobs=-1)

grid_rfc_auc.fit(X_sample, y_sample)

'''
Logistic Regression
'''

lr_pipe = Pipeline(
    [
     ('processor', preprocessor),
     ('best', SelectKBest(score_func=chi2)),
     ('lr', LogisticRegression(solver='liblinear', max_iter=10000))
    ]
)

parametters_lr = {
    'best__k':[10, 50, 100],
    'lr__penalty':['l2', 'l1'],
    'lr__C':[100, 10, 1.0, 0.1, 0.01]
}

grid_lr_auc = GridSearchCV(lr_pipe, 
                           param_grid=parametters_lr, 
                           scoring='roc_auc', 
                           n_jobs=-1)

grid_lr_auc.fit(X_sample, y_sample)


'''
Gradient Boosted Decision Trees 
'''

gbdt_pipe = Pipeline(
    [
     ('processor', preprocessor),
     ('best', SelectKBest(score_func=chi2)),
     ('gbdt', GradientBoostingClassifier(random_state=0))
    ]
)

parametters_gbdt = {
    'best__k':[10, 50, 100],
    'gbdt__n_estimators':[50, 100, 150],
    'gbdt__learning_rate':[0.001, 0.01, 0.1],
    'gbdt__max_depth':[2, 3, 4, 5]
}

grid_gbdt_auc = GridSearchCV(gbdt_pipe, 
                             param_grid=parametters_gbdt, 
                             scoring='roc_auc', 
                             n_jobs=-1)

grid_gbdt_auc.fit(X_sample, y_sample)

'''
Resume to choose model
'''

resume = pd.DataFrame({'SVC':[grid_svc_auc.best_params_, grid_svc_auc.best_score_],
                       'RFC':[grid_rfc_auc.best_params_, grid_rfc_auc.best_score_],
                       'LR':[grid_lr_auc.best_params_, grid_lr_auc.best_score_],
                       'GBDT':[grid_gbdt_auc.best_params_, grid_gbdt_auc.best_score_]}, 
                      index=['Params', 'AUC']).T
resume
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Params</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SVC</th>
      <td>{'best__k': 10, 'svc__C': 0.1, 'svc__gamma': 0.1}</td>
      <td>0.559088</td>
    </tr>
    <tr>
      <th>RFC</th>
      <td>{'best__k': 10, 'rfc__max_depth': 5, 'rfc__max...</td>
      <td>0.789177</td>
    </tr>
    <tr>
      <th>LR</th>
      <td>{'best__k': 100, 'lr__C': 100, 'lr__penalty': ...</td>
      <td>0.738138</td>
    </tr>
    <tr>
      <th>GBDT</th>
      <td>{'best__k': 10, 'gbdt__learning_rate': 0.01, '...</td>
      <td>0.804952</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
We choose the Gradient Boosted Decision Trees model with auc=0.805 with params:
{'best__k': 10,
 'gbdt__learning_rate': 0.01,
 'gbdt__max_depth': 5,
 'gbdt__n_estimators': 50}
'''

'''
We train a new model
'''

model = Pipeline(
    [
     ('processor', preprocessor),
     ('best', SelectKBest(score_func=chi2, k=10)),
     ('gbdt', GradientBoostingClassifier(
         learning_rate=0.01,
         max_depth=5,
         n_estimators=50,
         random_state=0
     )
     ) 
     ]
)                     
    
model.fit(X_train, y_train)

AUC_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
```


```python
print ('The AUC score is: ', AUC_score)
```

    The AUC score is:  0.7930930595118645
    

Now we can predict the target values from the test data. Let's look at the probabilities of predicting a specific ticket.


```python
original_data_test = pd.read_csv('datasets/test.csv')

probability = pd.Series(model.predict_proba(data_test)[:,0], 
                        original_data_test['ticket_id'], 
                        dtype='float32')

probability.head()
```




    ticket_id
    284932    0.935593
    285362    0.950084
    285361    0.935593
    285338    0.935593
    285346    0.935593
    dtype: float32


