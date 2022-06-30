# Document Similarity

In this project we are going to train a classifier to find if two texts are similar or not.


```python
# -*- coding: utf-8 -*-
'''
Libraries
'''

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from nltk import pos_tag, word_tokenize, WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer

nltk.download('words')

'''
We define the functions that we are going to use to process 
the texts from the NLTK library.
'''

def convert_tag(tag):
    '''
    This function convert the tag given by nltk.pos_tag 
    to the tag used by wordnet.synsets
    '''
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    '''
    Returns a list of synsets in document.

    Args:
        doc: string to be converted

    Returns:
        list of synsets
    '''
    
    lemma = WordNetLemmatizer() # Lemmatizer object
    
    tokens = word_tokenize(doc) # We tokenize the document
    
    words_tagged = pos_tag(tokens) # Token tagging
    
    answer = [
        (lemma.lemmatize(k[0]), convert_tag(k[1])) 
        for k in words_tagged
    ]    
    output = [
        wn.synsets(t[0], pos=t[1])[0] 
        for t in answer 
        if len(wn.synsets(t[0], pos=t[1])) > 0
    ] 
                
    return output


def similarity_score(s1, s2):
    '''
    This function calculate the normalized similarity score of s1 onto s2

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2
    ''' 
    
    answer = [
        np.max([
            k.path_similarity(v) 
            for v in s2 
            if k.path_similarity(v)!=None
        ]) 
        for k in s1 
        if len([
            k.path_similarity(v) for v 
            in s2 if k.path_similarity(v)!=None
        ])!=0
    ]
    
    output = np.mean(answer)
    
    return output


def document_path_similarity(doc1, doc2):
    '''
    This function finds the symmetrical similarity between doc1 and doc2
    '''

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1))/2
```

We are going to provide labels for a dataset with twenty pairs of documents 
by calculating the similarity of each pair using document_path_similarity. 
If the score is greater than 0.75, it is a paraphrase (positive class 1), 
otherwise the label is not a paraphrase (negative class 0).

We will train a Support vector machine classifier to try to predict the 
class of a test set


```python
paraphrases = pd.read_csv('datasets/paraphrases.csv') # Dataset  
paraphrases.head()
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
      <th>Quality</th>
      <th>D1</th>
      <th>D2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Ms Stewart, the chief executive, was not expec...</td>
      <td>Ms Stewart, 61, its chief executive officer an...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>After more than two years' detention under the...</td>
      <td>After more than two years in detention by the ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>"It still remains to be seen whether the reven...</td>
      <td>"It remains to be seen whether the revenue rec...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>And it's going to be a wild ride," said Allan ...</td>
      <td>Now the rest is just mechanical," said Allan H...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>The cards are issued by Mexico's consulates to...</td>
      <td>The card is issued by Mexico's consulates to i...</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
We clean and prepare the dataset We use the function document_path_similarity() 
to find the similarity between D1 and D2 and asign target value.
'''

data = list(
    zip(
        paraphrases['D1'].values, paraphrases['D2'].values
    )
)
    
paraphrases['target'] = pd.DataFrame(
    {'target':[np.where(document_path_similarity(k[0], k[1])<0.75, 0, 1) for k in data]}
)
    
df = paraphrases.drop('Quality', axis=1)
        
df[['D1', 'D2']] = df[['D1', 'D2']].apply(lambda x: x.apply(lambda y: y.lower()))
    
df[['D1', 'D2']] = df[['D1', 'D2']].apply(lambda x: x.str.replace(r'[_-]+', ' ', regex=True))
    
X = df[['D1', 'D2']]
y = df['target'].apply(lambda x: int(x))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```




    0.8



We design a pipeline to vectorize the texts and train an SVC classifier. We use GridSearch to optimize the model with the best parameters and based on the accuracy score because it is a small dataset and this score gives us enough information.


```python
model = Pipeline(
    [
    ('v1', ColumnTransformer(
        [
            ('step1', TfidfVectorizer(), 'D1'),
            ('step2', TfidfVectorizer(), 'D2')
        ]
        )
        ),
    ('svc', SVC())
    ]
)
    
    
parametters_svc = {
    'v1__step1__min_df':[1, 2, 3, 4, 5],
    'v1__step2__min_df':[1, 2, 3, 4, 5],
    'svc__C':[0.001, 0.1, 1, 10, 100],
    'svc__gamma':[0.001, 0.01, 0.1, 1, 10, 100]
}
    
grid = GridSearchCV(model, param_grid=parametters_svc, scoring='accuracy', n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
    
y_predict = grid.predict(X_test)
    
acc = accuracy_score(y_test, y_predict)

print ('Accuracy score: ', acc)
```

    Accuracy score:  0.8
    
