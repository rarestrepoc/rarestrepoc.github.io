# Salary Prediction

In this project we are going to work with a company's email network where each node corresponds to a person at the company, and each edge indicates that at least one email has been sent between two people.

The network also contains the node attributes `Department` and `ManagementSalary`.

`Department` indicates the department in the company which the person belongs to, and `ManagementSalary` indicates whether that person is receiving a management position salary.

Some of the ManagementSalary values are missing, so we want to predict whether or not these individuals are receiving a management position salary. 

We will create a matrix of node features using networkx and training a classifier model.

### Note

Due to the migration of NetworkX from v1.x to v2.x, where there was an incompatible change to their data structures, we go to read the original file named "'email_prediction.txt'" it in with v1.x installed and write a file with the node and edge information. After, we can read that into a config with v2.x installed and then add those nodes and edges to a fresh graph.

The code in NetworkX v1.x to do that:

```python 
G = nx.read_gpickle('datasets/email_prediction.txt')

file = open('datasets/myfile.txt', 'w')

with open('datasets/filename.pickle', 'wb') as handle:
    pickle.dump([G.nodes(data=True), G.edges(data=True)], handle) 
```


```python
import networkx as nx
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


'''
Now we can read this new corrected file and read the individual 
nodes and edges.

We use NetworkX v2.8.4
'''

with open('datasets/filename.pickle', 'rb') as handle:
    nodes, edges = pickle.load(handle)
    
G = nx.Graph()  
G.add_nodes_from(nodes)  
G.add_edges_from(edges)

list(G.nodes().items())[0:5]
```




    [(0, {'Department': 1, 'ManagementSalary': 0.0}),
     (1, {'Department': 1, 'ManagementSalary': nan}),
     (2, {'Department': 21, 'ManagementSalary': nan}),
     (3, {'Department': 21, 'ManagementSalary': 1.0}),
     (4, {'Department': 21, 'ManagementSalary': 1.0})]




```python
'''
We are going to create a dataframe where we store the nodes and we will 
create 3 characteristics: node attribute 'department', the degree of 
the node and the Clustering Coefficient of the node.
'''

gn = G.nodes()

df = pd.DataFrame(index=gn.keys())

df['department'] = pd.Series([gn[v]['Department'] for v in df.index])
df['target'] = pd.Series([gn[v]['ManagementSalary'] for v in df.index])
df['degree'] = pd.Series([k[1] for k in G.degree()]) # degree
df['clustering'] = pd.Series(nx.clustering(G)) # Clustering
    
df.head()
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
      <th>department</th>
      <th>target</th>
      <th>degree</th>
      <th>clustering</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>44</td>
      <td>0.276423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>NaN</td>
      <td>52</td>
      <td>0.265306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>NaN</td>
      <td>95</td>
      <td>0.297803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>1.0</td>
      <td>71</td>
      <td>0.384910</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>1.0</td>
      <td>96</td>
      <td>0.318691</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Let's create the datasets to train the model
'''

X = df[df['target'].eq(1) | df['target'].eq(0)].drop('target', axis=1)
y = df[df['target'].eq(1) | df['target'].eq(0)]['target']
        
X_target = df[df['target'].ne(1) & df['target'].ne(0)].drop('target', axis=1)
   
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

'''
We are going to use the RandomForestClassifier and 
GridSearch model to optimize it based on the AUC score.
'''
    
model = RandomForestClassifier(n_jobs=-1, random_state=0)
    
parametters_rfc = {
    'n_estimators':[50, 100, 150], 
    'max_features':[1, 2], 
    'max_depth':[2, 3, 4, 5]
}
    
grid_rfc =  GridSearchCV(model, 
                         param_grid=parametters_rfc, 
                         scoring='roc_auc', cv=3)

grid_rfc.fit(X_train, y_train)
    
AUC_score = roc_auc_score(y_test, grid_rfc.predict_proba(X_test)[:,1])

print ('AUC score with Random Forest Classifier model: ', AUC_score)    
```

    AUC score with Random Forest Classifier model:  0.8248427672955975
    

Now we want to predict whether or not these individuals (nodes where ManagementSalary values are missing) are receiving a management position salary. This is the probability of predicting the target value for each node.


```python
pd.Series(index=X_target.index, data=grid_rfc.predict_proba(X_target)[:,0])
```




    1       0.971918
    2       0.399799
    5       0.000820
    8       0.690199
    14      0.902779
              ...   
    992     0.991071
    994     0.991503
    996     0.993502
    1000    0.952938
    1001    0.930873
    Length: 252, dtype: float64



 # New Connections Prediction
 
Now we will predict future connections between employees of the network. The future connections information has been loaded into the variable future_connections from csv file. The index is a tuple indicating a pair of nodes that currently do not have a connection, and the Future Connection column indicates if an edge between those two nodes will exist in the future, where a value of 1.0 indicates a future connection.

We want to predict future connections between employees using a classifier. We go to create a matrix of features for the edges found in future_connections using networkx.

We want to predict future connections between employees using a classifier. We go to create a matrix of features for the edges found in future_connections using networkx.

Features:
- Jaccard Coefficient
- Common Neighbors
- Resource Allocation
- Adamic-Adar Index
- Pref. Attachment


```python
df = pd.read_csv('datasets/Future_Connections.csv', index_col=0, converters={0: eval})

'''
Common Neighbors
'''

df['Common Neighbors'] = df.index.map(
    lambda x: len(list(nx.common_neighbors(G, x[0], x[1])))
)

'''
Pref. Attachment
'''

df['Preferential Attachment'] = [
    i[2] for i in nx.preferential_attachment(G, df.index)
]

'''
Jaccard Coefficient
'''


df['Jaccard Coefficient'] = df.index.to_series().map(
    {
        (k[0], k[1]):k[2] for k in list(nx.jaccard_coefficient(G))
    }
)

'''
Resource Allocation
'''

df['Resource Allocation'] = df.index.to_series().map(
    {
        (k[0], k[1]):k[2] for k in list(nx.resource_allocation_index(G))
    }
)

'''
Adamic-Adar Index
'''

df['Adamic-Adar'] = df.index.to_series().map(
    {
        (k[0], k[1]):k[2] for k in list(nx.adamic_adar_index(G))
    }
)
    
df.head()
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
      <th>Future Connection</th>
      <th>Common Neighbors</th>
      <th>Preferential Attachment</th>
      <th>Jaccard Coefficient</th>
      <th>Resource Allocation</th>
      <th>Adamic-Adar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(6, 840)</th>
      <td>0.0</td>
      <td>9</td>
      <td>2070</td>
      <td>0.073770</td>
      <td>0.136721</td>
      <td>2.110314</td>
    </tr>
    <tr>
      <th>(4, 197)</th>
      <td>0.0</td>
      <td>2</td>
      <td>3552</td>
      <td>0.015504</td>
      <td>0.008437</td>
      <td>0.363528</td>
    </tr>
    <tr>
      <th>(620, 979)</th>
      <td>0.0</td>
      <td>0</td>
      <td>28</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(519, 872)</th>
      <td>0.0</td>
      <td>2</td>
      <td>299</td>
      <td>0.060606</td>
      <td>0.039726</td>
      <td>0.507553</td>
    </tr>
    <tr>
      <th>(382, 423)</th>
      <td>0.0</td>
      <td>0</td>
      <td>205</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can now train a logistic regression model using the data frame with the new features and optimizing it with GridSearch based on the AUC score


```python
X = df[df['Future Connection'].eq(1) | df['Future Connection'].eq(0)].drop('Future Connection', axis=1)
y = df[df['Future Connection'].eq(1) | df['Future Connection'].eq(0)]['Future Connection']


'''
X_target is the dataframe with Future Connection missing data.
'''

X_target = df[df['Future Connection'].ne(1) & df['Future Connection'].ne(0)].drop('Future Connection', axis=1)  
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
model = LogisticRegression(max_iter=10000)
    
parametters_lr = {
    'C':[100, 10, 1.0, 0.1, 0.01]
}
    
grid_lr =  GridSearchCV(model, 
                        param_grid=parametters_lr, 
                        scoring='roc_auc', 
                        cv=3)

grid_lr.fit(X_train, y_train)

AUC_score = roc_auc_score(y_test, grid_lr.predict_proba(X_test)[:,1])

print ('AUC score with Logistic Regression model: ', AUC_score)  
```

    AUC score with Logistic Regression model:  0.9065970350007253
    

And now we can predict if there will be a connection between the workers represented by the nodes.
0 --> there will be no connection
1 --> there will be connection.


```python
pd.Series(index=X_target.index, data=grid_lr.predict(X_target))
```




    (107, 348)    0.0
    (542, 751)    0.0
    (20, 426)     1.0
    (50, 989)     0.0
    (942, 986)    0.0
                 ... 
    (165, 923)    0.0
    (673, 755)    0.0
    (939, 940)    0.0
    (555, 905)    0.0
    (75, 101)     0.0
    Length: 122112, dtype: float64


