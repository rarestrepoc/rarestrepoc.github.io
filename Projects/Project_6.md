# Text Mining 

We have a dataset made up of emails where we want to see if the email is spam (1) or not (0).


```python
'''
Libraries and dataset
'''

# -*- coding: utf-8 -*-
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

spam_data = pd.read_csv('datasets/spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)

spam_data.head()
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
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ok lar... Joking wif u oni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

'''
We are going to create 3 new features from the text:

- The length of document (number of characters)
- Number of digits per document
- Number of non-word characters (anything other than a letter, digit or underscore.)
'''


def len_doc(X):
    
    '''
    This function takes as input a column of a dataframe and will 
    return an csr matrix with the lengths of the text of each row.
    '''
    
    return csr_matrix(X.apply(lambda x: len(x))).T

def dig_per_doc(X):
    
    '''
    This function takes a column of a dataframe as input and will return 
    an csr matrix with the number of digits contained in the text of each row.
    '''
    
    pattern = r'\d*'
    
    return csr_matrix(X.str.findall(pattern).apply(lambda x: np.sum([len(k) for k in x if len(k)!=0]))).T

def non_words_char(X):
    
    '''
    This function takes a column of a dataframe as input and will return an 
    array with the number of non-word characters (anything other than a letter, 
    digit or underscore) contained in the text of each row.
    '''
    
    pattern = r'\W*'
    
    return csr_matrix(X.str.findall(pattern).apply(lambda x: np.sum([len(k) for k in x if len(k)!=0]))).T
```

We are going to train and find the best parameters of 3 different classification models: SVC, Multinomial Naïve Bayes and Logistic Regression. We are going to train and find the best parameters of 3 different classification models: SVC, Multinomial Naïve Bayes and Logistic Regression with AUC score as evaluation method. For each model we design a pipeline where we vectorize each text with the method TF-IDF using TfidfVectorizer and add the 3 features already mentioned using the above functions using FeatureUnion.


```python
'''
Kernelized Support Vector Machine with RBF kernel
'''
        
model_SVC = Pipeline(
    [
     (
      'features', FeatureUnion(
         [
             # Vectorize vectorize input data (emails)
             ('CV', TfidfVectorizer()), 
             
             # Create the new features
             ('len', FunctionTransformer(len_doc)),
             ('dig', FunctionTransformer(dig_per_doc)),
             ('non_words', FunctionTransformer(non_words_char)),
             ]
         )
         ),
     ('svc', SVC(kernel='rbf', random_state=0))
     ]
    ) 

parametters_svc = {
    'features__CV__min_df':[3, 4, 5],
    'features__CV__ngram_range':[(1,3), (2,5)],
    'svc__C':[0.001, 0.1, 1, 10, 100],
    'svc__gamma':[0.001, 0.01, 0.1, 1, 10, 100]
    }

grid_svc = GridSearchCV(model_SVC, 
                        param_grid=parametters_svc, 
                        scoring='roc_auc',
                        n_jobs=-1)

grid_svc.fit(X_train, y_train)

'''
Multinomial Naïve Bayes
'''

model_MultinomialNB = Pipeline(
    [
     (
      'features', FeatureUnion(
         [
             ('CV', TfidfVectorizer()), 
             ('len', FunctionTransformer(len_doc)),
             ('dig', FunctionTransformer(dig_per_doc)),
             ('non_words', FunctionTransformer(non_words_char)),
             ]
         )
         ),
     ('NB', MultinomialNB())
     ]
    ) 

parametters_MultinomialNB = {
    'features__CV__min_df':[3, 4, 5],
    'features__CV__ngram_range':[(1,3), (2,5)],
    'NB__alpha':[0.001, 0.1, 1, 10, 100]
    }

grid_NB = GridSearchCV(model_MultinomialNB, 
                        param_grid=parametters_MultinomialNB, 
                        scoring='roc_auc',
                        n_jobs=-1)

grid_NB.fit(X_train, y_train)

'''
Logistic Regression
'''

model_lr = Pipeline(
    [
     (
      'features', FeatureUnion(
         [
             ('CV', TfidfVectorizer()), 
             ('len', FunctionTransformer(len_doc)),
             ('dig', FunctionTransformer(dig_per_doc)),
             ('non_words', FunctionTransformer(non_words_char)),
             ]
         )
         ),
     ('LR', LogisticRegression(max_iter=10000))
     ]
    ) 

parametters_lr = {
    'features__CV__min_df':[3, 4, 5],
    'features__CV__ngram_range':[(1,3), (2,5)],
    'LR__C':[100, 10, 1.0, 0.1, 0.01]
    }

grid_LR = GridSearchCV(model_lr, 
                        param_grid=parametters_lr, 
                        scoring='roc_auc',
                        n_jobs=-1)

grid_LR.fit(X_train, y_train)

'''
Let's see which model obtained the best AUC score.
'''

resume = pd.DataFrame({'SVC':[grid_svc.best_params_, grid_svc.best_score_],
                       'NB':[grid_NB.best_params_, grid_NB.best_score_],
                       'LR':[grid_LR.best_params_, grid_LR.best_score_]}, 
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
      <td>{'features__CV__min_df': 5, 'features__CV__ngr...</td>
      <td>0.97782</td>
    </tr>
    <tr>
      <th>NB</th>
      <td>{'NB__alpha': 0.001, 'features__CV__min_df': 3...</td>
      <td>0.977138</td>
    </tr>
    <tr>
      <th>LR</th>
      <td>{'LR__C': 10, 'features__CV__min_df': 4, 'feat...</td>
      <td>0.994043</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
We chose the Logistic Regression model with AUC = 0.994043.
'''

AUC_score = roc_auc_score(y_test, grid_LR.predict_proba(X_test)[:,1])

print ('AUC score with Logistic Regression model: ', AUC_score)
```

    AUC score with Logistic Regression model:  0.9978566456716976
    

We obtain an AUC of 0.998 using our trained model with the best parameters on the test set, so our model performs very well in classifying emails as spam and not spam.
