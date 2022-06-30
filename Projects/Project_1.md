# Project Data Analysis

In this project we read in a file of metropolitan regions and associated sports teams from assets/wikipedia_data.html. Each of these regions may have one or more teams from the "Big 4": NFL (football, in assets/nfl.csv), MLB (baseball, in assets/mlb.csv), NBA (basketball, in assets/nba. csv or NHL (hockey, in assets/nhl.csv).

Later we clean, normalize and join the datasets in a main dataset to answer a series of questions from the perspective of the metropolitan region.


```python
'''
We import the necessary libraries.
'''

import pandas as pd
import numpy as np
import re
from scipy import stats
from scipy.stats import ttest_ind
import unicodedata

'''
This feature allows you to clean and separate a single column containing the metro area 
and equipment into two.
'''

def func_col(column):
    def separate(cadena, tp = None):
        pattern = '([A-Za-z]*\.* *[A-Za-z]*) +([A-Za-z0-9]+)\** *\(*\w*\)*'
        cadena = unicodedata.normalize('NFKD', cadena)
        state = re.findall(pattern, cadena)[0]
        if tp == 0:
            return state[0]
        elif tp == 1:
            return state[1]
        elif tp == None:
            return state

    column = column.set_index('year').loc[2018,]
    column['Metropolitan area'] = column['team'].apply(lambda x: separate(x, tp = 0))
    column['Team'] = column['team'].apply(lambda x: separate(x, tp = 1))
    column = column.drop('team', axis=1)
    
    return column

'''
This function allows to standardize some names to correctly join the datasets.
'''

def corr(x):
    names = {'New York City':'New York',
             'San Francisco Bay Area':'San Francisco',
             'Dallas–Fort Worth':'Dallas',
             'Washington, D.C.':'Washington',
             'Minneapolis–Saint Paul':'Minnesota',
             'Miami–Fort Lauderdale':'Miami',
             'Tampa Bay Area':'Tampa Bay',
             'Phoenix':'Arizona',
             'Salt Lake City':'Utah'}
    if names.get(x, None) == None:
        return x
    else:
        return names.get(x, None)

'''
We first read the html file from Wikipedia that contains the information on population by metropolitan area 
and sports teams by metropolitan area.
'''    
    
html = (pd.read_html('datasets/wikipedia_data.html')[1].rename({'Population (2016 est.)[8]':'Population'}, axis=1))

```

Clean html DataFrame


```python
html = (pd.read_html('datasets/wikipedia_data.html')[1].rename({'Population (2016 est.)[8]':'Population'}, axis=1))
html = html.iloc[:-1,[0,3,5,6,7,8]]
html['Metropolitan area'] = html['Metropolitan area'].apply(lambda x: corr(x))
html.head()
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
      <th>Metropolitan area</th>
      <th>Population</th>
      <th>NFL</th>
      <th>MLB</th>
      <th>NBA</th>
      <th>NHL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New York</td>
      <td>20153634</td>
      <td>GiantsJets[note 1]</td>
      <td>YankeesMets[note 2]</td>
      <td>KnicksNets</td>
      <td>RangersIslandersDevils[note 3]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Los Angeles</td>
      <td>13310447</td>
      <td>RamsChargers[note 4]</td>
      <td>DodgersAngels</td>
      <td>LakersClippers</td>
      <td>KingsDucks</td>
    </tr>
    <tr>
      <th>2</th>
      <td>San Francisco</td>
      <td>6657982</td>
      <td>49ersRaiders[note 6]</td>
      <td>GiantsAthletics</td>
      <td>Warriors</td>
      <td>Sharks[note 7]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chicago</td>
      <td>9512999</td>
      <td>Bears[note 8]</td>
      <td>CubsWhite Sox</td>
      <td>Bulls[note 9]</td>
      <td>Blackhawks</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dallas</td>
      <td>7233323</td>
      <td>Cowboys</td>
      <td>Rangers</td>
      <td>Mavericks</td>
      <td>Stars</td>
    </tr>
  </tbody>
</table>
</div>



We can first calculate the correlation of the win/loss ratio with the population of the city for the NHL using 2018 data.


```python
'''
Read NHL csv
'''

NHL = pd.read_csv('datasets/nhl.csv')
NHL.head()
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
      <th>team</th>
      <th>GP</th>
      <th>W</th>
      <th>L</th>
      <th>OL</th>
      <th>PTS</th>
      <th>PTS%</th>
      <th>GF</th>
      <th>GA</th>
      <th>SRS</th>
      <th>SOS</th>
      <th>RPt%</th>
      <th>ROW</th>
      <th>year</th>
      <th>League</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>Atlantic Division</td>
      <td>2018</td>
      <td>NHL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tampa Bay Lightning*</td>
      <td>82</td>
      <td>54</td>
      <td>23</td>
      <td>5</td>
      <td>113</td>
      <td>.689</td>
      <td>296</td>
      <td>236</td>
      <td>0.66</td>
      <td>-0.07</td>
      <td>.634</td>
      <td>48</td>
      <td>2018</td>
      <td>NHL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Boston Bruins*</td>
      <td>82</td>
      <td>50</td>
      <td>20</td>
      <td>12</td>
      <td>112</td>
      <td>.683</td>
      <td>270</td>
      <td>214</td>
      <td>0.62</td>
      <td>-0.07</td>
      <td>.610</td>
      <td>47</td>
      <td>2018</td>
      <td>NHL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Toronto Maple Leafs*</td>
      <td>82</td>
      <td>49</td>
      <td>26</td>
      <td>7</td>
      <td>105</td>
      <td>.640</td>
      <td>277</td>
      <td>232</td>
      <td>0.49</td>
      <td>-0.06</td>
      <td>.567</td>
      <td>42</td>
      <td>2018</td>
      <td>NHL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Florida Panthers</td>
      <td>82</td>
      <td>44</td>
      <td>30</td>
      <td>8</td>
      <td>96</td>
      <td>.585</td>
      <td>248</td>
      <td>246</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>.537</td>
      <td>41</td>
      <td>2018</td>
      <td>NHL</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Clean NHL dataframe
'''

NHL = NHL[NHL['team'].ne('Atlantic Division') & 
          NHL['team'].ne('Metropolitan Division') &
          NHL['team'].ne('Central Division') &
          NHL['team'].ne('Pacific Division')]
    
NHL = func_col(NHL) # Here we separate the name of the metropolitan area and the name of the team

'''
We have to manually change some names in the dataframe
'''
NHL.iloc[2,-1] = 'Maple Leafs'
NHL.iloc[2,-2] = 'Toronto'
NHL.iloc[4,-1] = 'Red Wings'
NHL.iloc[4,-2] = 'Detroit'
NHL.iloc[11,-1] = 'Blue Jackets'
NHL.iloc[11,-2] = 'Columbus'
NHL.iloc[23,-1] = 'Golden Knights'
NHL.iloc[23,-2] = 'Las Vegas'
NHL.iloc[12,-2] = 'New York'
NHL.iloc[3,-2] = 'Miami'
NHL.iloc[19,-2] = 'Denver'
NHL.iloc[24,-2] = 'Los Angeles'
NHL.iloc[25,-2] = 'San Francisco'
NHL.iloc[13,-2] = 'Raleigh'
NHL[['L', 'W']] = NHL[['L', 'W']].apply(lambda x: x.apply(lambda y: float(y)))

'''
Calc Win/Loss ratio
'''
NHL['W-L%'] = NHL['W']/(NHL['W'] + NHL['L'])
NHL = NHL.groupby('Metropolitan area').agg({'W-L%':np.mean})

'''
We prepare the html dataframe for this specific case
'''

htmlQ1 = html[html.set_index('Metropolitan area').index.isin(NHL.index)].set_index('Metropolitan area').sort_index()
htmlQ1 = htmlQ1['Population'].apply(lambda x: float(x))

'''
Merge both dataframes
'''

df_Q1 = pd.merge(NHL, htmlQ1, how = 'left', left_index=True, right_index=True)
df_Q1 = df_Q1.rename(index={'Arizona':'Phoenix','Carolina':'Charlotte'}).sort_index()

'''
We use the Pearson Correlation Coefficient to find the correlation
'''

population_by_region_Q1 = df_Q1['Population']
win_loss_by_region_Q1 = df_Q1['W-L%']

answer_1 = stats.pearsonr(population_by_region_Q1, win_loss_by_region_Q1)[0]
print ('Correlation between w/l ratio and population by region:', answer_1)
```

    Correlation between w/l ratio and population by region 0.012486162921209907
    

We obtain a correlation coefficient of 0.012; we can say that there is no correlation between the win/loss ratio and the population. Now we answer the same question with the NBA.


```python
'''
Read NBA csv
'''

NBA = pd.read_csv('datasets/nba.csv')
NBA = func_col(NBA)

'''
We have to manually change some names in the dataframe
'''

NBA.iloc[29, -2] = 'Arizona'
NBA.iloc[17, -1] = 'Trail Blazers'
NBA.iloc[17, -2] = 'Portland'
NBA.iloc[16, -2] = 'San Francisco'
NBA.iloc[11, -2] = 'New York'
NBA.iloc[4, -2] = 'Indianapolis'

'''
Calc Win/Loss ratio
'''

NBA['W/L%'] = NBA['W/L%'].apply(lambda x: float(x))
NBA = NBA.groupby('Metropolitan area').agg({'W/L%':np.mean}).sort_index()

'''
We prepare the html dataframe for this specific case
'''

htmlQ2 = html[html.set_index('Metropolitan area').index.isin(NBA.index)].set_index('Metropolitan area').sort_index()
htmlQ2 = htmlQ2['Population'].apply(lambda x: float(x))

'''
Merge both dataframes
'''

df_Q2 = pd.merge(NBA, htmlQ2, how = 'left', left_index=True, right_index=True)
df_Q2 = df_Q2.rename(index={'Arizona':'Phoenix','Carolina':'Charlotte'}).sort_index()

population_by_region_Q2 = df_Q2['Population'] 
win_loss_by_region_Q2 = df_Q2['W/L%'] 

answer_2 = stats.pearsonr(population_by_region_Q2, win_loss_by_region_Q2)[0]
print ('Correlation between w/l ratio and population by region:', answer_2)
```




    -0.17636350642182938



We obtain a correlation coefficient of -0.18; we can say that there is weak correlation between the win/loss ratio and the population. 

We can do the same process for MLB and NHL.


```python
'''
For MLB
'''

MLB = pd.read_csv('datasets/mlb.csv')
MLB = func_col(MLB)
MLB.iloc[0,-2] = 'Boston'
MLB.iloc[8,-2] = 'Chicago'
MLB.iloc[0,-1] = 'Red Sox'
MLB.iloc[8,-1] = 'White Sox'
MLB.iloc[3,-1] = 'Blue Jays'
MLB.iloc[3,-2] = 'Toronto'
MLB.iloc[11,-2] = 'San Francisco'
MLB.iloc[14,-2] = 'Dallas'
MLB.iloc[26,-2] = 'Denver'
MLB[['L', 'W']] = MLB[['L', 'W']].apply(lambda x: x.apply(lambda y: float(y)))
MLB['W-L%'] = MLB['W']/(MLB['W'] + MLB['L'])
MLB = MLB.groupby('Metropolitan area').agg({'W-L%':np.mean}).sort_index()

htmlQ3 = html[html.set_index('Metropolitan area').index.isin(MLB.index)].set_index('Metropolitan area').sort_index()
htmlQ3 = htmlQ3['Population'].apply(lambda x: float(x))

df_Q3 = pd.merge(MLB, htmlQ3, how = 'left', left_index=True, right_index=True)
df_Q3 = df_Q3.rename(index={'Arizona':'Phoenix','Carolina':'Charlotte'}).sort_index()

population_by_region_Q3 = df_Q3['Population']
win_loss_by_region_Q3 = df_Q3['W-L%']

answer_3 = stats.pearsonr(population_by_region_Q3, win_loss_by_region_Q3)[0]

'''
For NFL
'''

NFL = pd.read_csv('datasets/nfl.csv')
    
NFL = NFL[NFL['team'].ne('AFC East') &
          NFL['team'].ne('AFC North') &
          NFL['team'].ne('AFC South') &
          NFL['team'].ne('AFC West') &
          NFL['team'].ne('NFC East') &
          NFL['team'].ne('NFC North') &
          NFL['team'].ne('NFC South') &
          NFL['team'].ne('NFC West')] 

NFL = func_col(NFL)

NFL.iloc[10,-2] = 'Nashville'
NFL.iloc[15,-2] = 'San Francisco'
NFL.iloc[0,-2] = 'Boston'
NFL.iloc[-7,-2] = 'Charlotte'

NFL['W-L%'] = NFL['W-L%'].apply(lambda x: float(x))
NFL = NFL.groupby('Metropolitan area').agg({'W-L%':np.mean}).sort_index()

htmlQ4 = html[html.set_index('Metropolitan area').index.isin(NFL.index)].set_index('Metropolitan area').sort_index()
htmlQ4 = htmlQ4['Population'].apply(lambda x: float(x))

df_Q4 = pd.merge(NFL, htmlQ4, how = 'left', left_index=True, right_index=True)
df_Q4 = df_Q4.rename(index={'Arizona':'Phoenix','Carolina':'Charlotte'}).sort_index()

population_by_region_Q4 = df_Q4['Population']
win_loss_by_region_Q4 = df_Q4['W-L%']

answer_4 = stats.pearsonr(population_by_region_Q4, win_loss_by_region_Q4)[0]

print ('Correlation between MLB w/l ratio and population by region:', answer_3,
       '\nCorrelation between NFL w/l ratio and population by region:', answer_4)

```

    Correlation between MLB w/l ratio and population by region: 0.15027698302669307 
    Correlation between NFL w/l ratio and population by region: 0.004282141436393017
    

As we can see, the w/l ratio of MLB has a weak positive correlation and NHL has no correlation with population by metropolitan area.

Now we would like to explore the hypothesis that given that an area has two sports teams in different sports, those teams will perform the same within their respective sports. We can do this with a series of paired t-tests between all pairs of sports with an significance level of 0,05. 


```python
sports = ['NFL', 'NBA', 'MLB', 'NHL']

'''
We create a df where we join the dataframes that contain the w/l ratios for each league, 
this to do the calculations in a single step. We only include for each sport the cities 
that have teams that participate in that sport.
'''

df_sports = pd.merge(df_Q4.rename(columns={'W-L%':'NFL'})['NFL'], 
                     df_Q2.rename(columns={'W/L%':'NBA'})['NBA'], 
                     how = 'outer', 
                     right_index=True, 
                     left_index=True)

df_sports = pd.merge(df_sports, 
                     df_Q3.rename(columns={'W-L%':'MLB'})['MLB'], 
                     how = 'outer', 
                     right_index=True, 
                     left_index=True)

df_sports = pd.merge(df_sports, 
                     df_Q1.rename(columns={'W-L%':'NHL'})['NHL'], 
                     how = 'outer', 
                     right_index=True, 
                     left_index=True)

'''
We create a list of dictionaries where each dictionary has a sport 
as its key and the calculation of each paired t-test as its value. 
'''

dic_answer = [
    {
        k:stats.ttest_rel(df_sports[['{}'.format(v), '{}'.format(k)]]['{}'.format(v)], 
                          df_sports[['{}'.format(v), '{}'.format(k)]]['{}'.format(k)], 
                          nan_policy='omit')[1] 
        for k in sports if v!=k
    } 
    for v in sports
]

'''
We create a dataframe with the list of dictionaries to visualize the results.
'''

p_values = pd.DataFrame(dic_answer, index=sports).sort_values(by=sports).T.sort_values(by=sports)
p_values
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
      <th>NHL</th>
      <th>MLB</th>
      <th>NBA</th>
      <th>NFL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NHL</th>
      <td>NaN</td>
      <td>0.000708</td>
      <td>0.022386</td>
      <td>0.030318</td>
    </tr>
    <tr>
      <th>MLB</th>
      <td>0.000708</td>
      <td>NaN</td>
      <td>0.948478</td>
      <td>0.803030</td>
    </tr>
    <tr>
      <th>NBA</th>
      <td>0.022386</td>
      <td>0.948478</td>
      <td>NaN</td>
      <td>0.937509</td>
    </tr>
    <tr>
      <th>NFL</th>
      <td>0.030318</td>
      <td>0.803030</td>
      <td>0.937509</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Our null hypothesis is that since an area has two sports teams in different sports, those teams will perform the same within their respective sports. With a significance level of 0.05, we can reject the null hypothesis with the MLB-NHL, NBA-NHL, and NFL-NHL team pairs since their p-value is less than the significance level.


```python

```
