---
title: "Modeling NBA salary with biological features"
excerpt: "Predicting salary without any performance statistics is really hard.  <br/><img src='/images/nba/dirk_pointing.jpg' width='533' height='300'>"
collection: portfolio
---
# Predicting NBA salary from biological stats.
I will follow the paradigm set out by ML lessons:
1. Find data sources (check! thanks Kaggle)
1. Explore and visualize the data
1. Clean data, 
1. Feature engineer
1. Additional data
1. Train models
1. Deploy best model.

Refs: https://www.ibm.com/downloads/cas/7109RLQM, coursera, 


## Imports and settings


```python
# Other
import os

#data science
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

# Visualization
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14,
                    'figure.figsize':(11,7)})

```


```python
def plot_metric_salary(metric, joined_data):
    joined_data_replace_nan = joined_data.fillna(0)
    data = joined_data_replace_nan[joined_data_replace_nan[metric] > 0]
    plt.scatter(x=data[metric], y=data['Log Salary (M $)'], s=50)
    plt.xlabel(metric)
    plt.ylabel('Log Salary (Millions, $)')
    plt.show()
```

## Find data sources
These are from kaggle: player salary and player bio stats


```python
player_salary = pd.read_csv(os.path.join('NBA_data','player_salary_2017','NBA_season1718_salary.csv'))
player_salary.tail()
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
      <th>Unnamed: 0</th>
      <th>Player</th>
      <th>Tm</th>
      <th>season17_18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>568</th>
      <td>569</td>
      <td>Quinn Cook</td>
      <td>NOP</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>569</th>
      <td>570</td>
      <td>Chris Johnson</td>
      <td>HOU</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>570</th>
      <td>571</td>
      <td>Beno Udrih</td>
      <td>DET</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>571</th>
      <td>572</td>
      <td>Joel Bolomboy</td>
      <td>MIL</td>
      <td>22248.0</td>
    </tr>
    <tr>
      <th>572</th>
      <td>573</td>
      <td>Jarell Eddie</td>
      <td>CHI</td>
      <td>17224.0</td>
    </tr>
  </tbody>
</table>
</div>



From https://www.statista.com/statistics/1009569/minimum-nba-salary/, the minimum salary is $815615. So let's elminate anyone with a salary below that. I'm not sure how they snuck in there!


```python
player_salary = player_salary[player_salary['season17_18'] >= 815615]
```


```python
player_bio_stats = pd.read_csv(os.path.join('NBA_data','player_measurements_1947-to-2017','player_measures_1947-2017.csv'))
player_bio_stats.head()
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
      <th>Player Full Name</th>
      <th>Birth Date</th>
      <th>Year Start</th>
      <th>Year End</th>
      <th>Position</th>
      <th>Height (ft 1/2)</th>
      <th>Height (inches 2/2)</th>
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>Hand Length (in inches)</th>
      <th>Hand Width (in inches)</th>
      <th>Weight (in lb)</th>
      <th>Body Fat (%)</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A.C. Green</td>
      <td>10/4/1963</td>
      <td>1986</td>
      <td>2001</td>
      <td>F-C</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>205.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>220.0</td>
      <td>NaN</td>
      <td>Oregon State University</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A.J. Bramlett</td>
      <td>1/10/1977</td>
      <td>2000</td>
      <td>2000</td>
      <td>C</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>208.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>227.0</td>
      <td>NaN</td>
      <td>University of Arizona</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A.J. English</td>
      <td>7/11/1967</td>
      <td>1991</td>
      <td>1992</td>
      <td>G</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>190.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>175.0</td>
      <td>NaN</td>
      <td>Virginia Union University</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A.J. Guyton</td>
      <td>2/12/1978</td>
      <td>2001</td>
      <td>2003</td>
      <td>G</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>185.4</td>
      <td>192.4</td>
      <td>247.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>180.0</td>
      <td>NaN</td>
      <td>Indiana University</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A.J. Hammons</td>
      <td>8/27/1992</td>
      <td>2017</td>
      <td>2017</td>
      <td>C</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>213.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>260.0</td>
      <td>NaN</td>
      <td>Purdue University</td>
    </tr>
  </tbody>
</table>
</div>



I want to limit my predictor data to match my salary data: Players active in 2017-2018 season\
From the first entry, A. C. Green, I can understand the convention for this dataset. A. C. Green's first season was the 1985-1986 season and his last season was the 2000-2001 season. So the `Year Start` and `Year End` columns use the later year of the season.

(Note, that's a super long career A. C.! https://en.wikipedia.org/wiki/A._C._Green)


```python
player_bio_1718 = player_bio_stats[
    (player_bio_stats['Year Start'] <= 2018)&(player_bio_stats['Year End'] >= 2018)
].reset_index(drop=True)
player_bio_1718.head()
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
      <th>Player Full Name</th>
      <th>Birth Date</th>
      <th>Year Start</th>
      <th>Year End</th>
      <th>Position</th>
      <th>Height (ft 1/2)</th>
      <th>Height (inches 2/2)</th>
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>Hand Length (in inches)</th>
      <th>Hand Width (in inches)</th>
      <th>Weight (in lb)</th>
      <th>Body Fat (%)</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaron Brooks</td>
      <td>1/14/1985</td>
      <td>2008</td>
      <td>2018</td>
      <td>G</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>182.9</td>
      <td>193.0</td>
      <td>238.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
      <td>2.7%</td>
      <td>University of Oregon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aaron Gordon</td>
      <td>9/16/1995</td>
      <td>2015</td>
      <td>2018</td>
      <td>F</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>205.7</td>
      <td>212.7</td>
      <td>266.7</td>
      <td>8.75</td>
      <td>10.5</td>
      <td>220.0</td>
      <td>5.1%</td>
      <td>University of Arizona</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abdel Nader</td>
      <td>9/25/1993</td>
      <td>2018</td>
      <td>2018</td>
      <td>F</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>198.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>230.0</td>
      <td>NaN</td>
      <td>Iowa State University</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adreian Payne</td>
      <td>2/19/1991</td>
      <td>2015</td>
      <td>2018</td>
      <td>F-C</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>208.3</td>
      <td>223.5</td>
      <td>276.9</td>
      <td>9.25</td>
      <td>9.5</td>
      <td>237.0</td>
      <td>7.6%</td>
      <td>Michigan State University</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Al Horford</td>
      <td>6/3/1986</td>
      <td>2008</td>
      <td>2018</td>
      <td>C-F</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>208.3</td>
      <td>215.3</td>
      <td>271.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>245.0</td>
      <td>9.1%</td>
      <td>University of Florida</td>
    </tr>
  </tbody>
</table>
</div>



## Explore and visualize the data



```python
player_salary['Salary (Million USD)'] = player_salary['season17_18']/1000000
#iq_hist = iqplot.histogram(data=player_salary,q='Salary (Million USD)', title='Histogram')
#iq_ecdf = iqplot.ecdf(data=player_salary,q='Salary (Million USD)',title='ECDF')
#bokeh.io.show(bokeh.layouts.gridplot([iq_hist, iq_ecdf], ncols=2))
fig, ax = plt.subplots(1,2, figsize=(11*2,7))
#fig, ax = plt.subplots(1,2)# figsize=(11,7))
plt.subplot(1,2,1)
plt.hist(player_salary['Salary (Million USD)'], align='right',rwidth=.95,)
plt.ylabel("Frequency")
plt.xlabel("Salary (Millions, $)")
plt.fig_size=(11,7)
plt.subplot(1,2,2)
plt.hist(player_salary['Salary (Million USD)'], align='right',rwidth=.95,cumulative=True,density=True,)
plt.ylabel("Cumulative probability")
plt.xlabel("Salary (Millions, $)")
plt.fig_size=(11,7)

plt.show()
```


    
![png](/images/NBA-bio/output_12_0.png)
    


These data are highly skewed, with a mean near \\$7.0 M/year and a standard deviation of \\$7.2 M/year, and a long tail towards the higher salaries


```python
player_salary['Salary (Million USD)'].mean(), player_salary['Salary (Million USD)'].std(), 
```




    (7.001891322851145, 7.336425350468712)



### Join the data

I have my predictor and response variables: biological stats and salary


```python
player_bio_1718['Player'] = player_bio_1718['Player Full Name']
joined_data = player_bio_1718.merge(player_salary, on="Player")
joined_data.head()
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
      <th>Player Full Name</th>
      <th>Birth Date</th>
      <th>Year Start</th>
      <th>Year End</th>
      <th>Position</th>
      <th>Height (ft 1/2)</th>
      <th>Height (inches 2/2)</th>
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>Hand Length (in inches)</th>
      <th>Hand Width (in inches)</th>
      <th>Weight (in lb)</th>
      <th>Body Fat (%)</th>
      <th>College</th>
      <th>Player</th>
      <th>Unnamed: 0</th>
      <th>Tm</th>
      <th>season17_18</th>
      <th>Salary (Million USD)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaron Brooks</td>
      <td>1/14/1985</td>
      <td>2008</td>
      <td>2018</td>
      <td>G</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>182.9</td>
      <td>193.0</td>
      <td>238.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
      <td>2.7%</td>
      <td>University of Oregon</td>
      <td>Aaron Brooks</td>
      <td>319</td>
      <td>MIN</td>
      <td>2116955.0</td>
      <td>2.116955</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aaron Gordon</td>
      <td>9/16/1995</td>
      <td>2015</td>
      <td>2018</td>
      <td>F</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>205.7</td>
      <td>212.7</td>
      <td>266.7</td>
      <td>8.75</td>
      <td>10.5</td>
      <td>220.0</td>
      <td>5.1%</td>
      <td>University of Arizona</td>
      <td>Aaron Gordon</td>
      <td>190</td>
      <td>ORL</td>
      <td>5504420.0</td>
      <td>5.504420</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abdel Nader</td>
      <td>9/25/1993</td>
      <td>2018</td>
      <td>2018</td>
      <td>F</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>198.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>230.0</td>
      <td>NaN</td>
      <td>Iowa State University</td>
      <td>Abdel Nader</td>
      <td>446</td>
      <td>BOS</td>
      <td>1167333.0</td>
      <td>1.167333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Al Horford</td>
      <td>6/3/1986</td>
      <td>2008</td>
      <td>2018</td>
      <td>C-F</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>208.3</td>
      <td>215.3</td>
      <td>271.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>245.0</td>
      <td>9.1%</td>
      <td>University of Florida</td>
      <td>Al Horford</td>
      <td>11</td>
      <td>BOS</td>
      <td>27734405.0</td>
      <td>27.734405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Al Jefferson</td>
      <td>1/4/1985</td>
      <td>2005</td>
      <td>2018</td>
      <td>C-F</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>208.3</td>
      <td>219.7</td>
      <td>279.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>289.0</td>
      <td>10.5%</td>
      <td>NaN</td>
      <td>Al Jefferson</td>
      <td>128</td>
      <td>IND</td>
      <td>9769821.0</td>
      <td>9.769821</td>
    </tr>
  </tbody>
</table>
</div>



For the player data, I'm going to look at Height (cm), Weight (lbs), age, ...


```python
# Clean this up by converting body fat from string to float
joined_data['Body Fat (%)'] = joined_data['Body Fat (%)'].str.rstrip('%').astype('float') / 100.0
```


```python
columns = player_bio_1718.columns
metrics = list(columns[7:-2])
# replace Nan with 0 for now, and
# neglect 0 values because these are physical measurements that should never be 0
joined_data_replace_nan = joined_data.fillna(0)
fig, ax = plt.subplots(7,2, figsize=(11*2,7*len(metrics)))
for count, metric in enumerate(metrics):
    data = joined_data_replace_nan[joined_data_replace_nan[metric] > 0]
    ax[count%len(metrics),0].hist(data[metric],rwidth=.95,)
    ax[count%len(metrics),0].set_xlabel(metric)
    ax[count%len(metrics),1].hist(data[metric],rwidth=.95,cumulative=True,density=True,)
    ax[count%len(metrics),1].set_xlabel(metric)
for hist in ax[:,0]:
    hist.set_ylabel('Frequency')
for CDF in ax[:,1]:
    CDF.set_ylabel('Cumulative probability')
```


    
![png](output_20_0.png)
    


These are looking pretty gausian, but I will have to scale them to make their values and ranges similar.

## Clean data and Feature Engineer

### Salary
The salarys appear roughly log-normal so I am going to transform them to make it look more Gaussian


```python
joined_data['Log Salary (M $)'] = np.log10(joined_data['Salary (Million USD)'])
plt.hist(joined_data['Log Salary (M $)'], align='right',rwidth=.95,)
plt.ylabel("Frequency")
plt.xlabel("Log 10 Salary (Millions, $)")
plt.show()
```


    
![png](/images/NBA-bio/output_24_0.png)
    


This shows it's more log uniform than log-normal. Nevertheless let's proceed:

Let's see how the biological stats compare to the Log10 Salary


```python
columns = player_bio_1718.columns
metrics = list(columns[7:-2])
scatter_list = []
# replace Nan with 0 for now, and
# neglect 0 values because these are physical measurements that should never be 0
joined_data_replace_nan = joined_data.fillna(0)
fig, ax = plt.subplots(4,2, figsize=(11*2,7*4))
for count, metric in enumerate(metrics):
    data = joined_data_replace_nan[joined_data_replace_nan[metric] > 0]
    ax[count//2, count%2].scatter(x=data[metric], 
                               y=data['Log Salary (M $)'], s=50)
    ax[count//2, count%2].set_xlabel(metric)
    ax[count//2, count%2].set_ylabel('Log Salary (Millions, $)')

```


    
![png](/images/NBA-bio/output_27_0.png)
    


These are looking generally pretty uniform, meaning salary is independant of most of these features.

There some data points at the extermes that stick out. Let's proceed.

### Create Features


```python
print('Total number of players: '+str(len(player_bio_1718)))
for column in player_bio_1718:
    print(player_bio_1718[column].isna().sum(), column)
```

    Total number of players: 471
    0 Player Full Name
    0 Birth Date
    0 Year Start
    0 Year End
    0 Position
    0 Height (ft 1/2)
    0 Height (inches 2/2)
    0 Height (in cm)
    177 Wingspan (in cm)
    177 Standing Reach (in cm)
    259 Hand Length (in inches)
    259 Hand Width (in inches)
    0 Weight (in lb)
    190 Body Fat (%)
    82 College
    0 Player
    

#### Predict wingspan

We're missing a fair amount of data in Body Fat, Wingspan, Reach, and Hand Length and Width. Can we reconstruct it from domain knowledge?


```python
data=player_bio_1718[player_bio_1718['Wingspan (in cm)'] > 0]
x_dim = 'Height (in cm)'
y_dim = 'Wingspan (in cm)'
plt.scatter(
    x=data[x_dim],
    y=data[y_dim],)
plt.xlabel(x_dim)
plt.ylabel(y_dim)

```




    Text(0, 0.5, 'Wingspan (in cm)')




    
![png](/images/NBA-bio/output_33_1.png)
    


We know that height can be used to predict wingspan fairly well in the general population, and the chart above is promising. Let's try it. 


```python
no_wingspan_data = player_bio_1718[player_bio_1718['Wingspan (in cm)'] > 0]
heights = no_wingspan_data['Height (in cm)']
wingspans = no_wingspan_data['Wingspan (in cm)']
regression = LinearRegression().fit(np.array(heights).reshape(-1,1), np.array(wingspans))
player_bio_1718['Wingspan predictions (in cm)'] = regression.predict(
    np.array(player_bio_1718['Height (in cm)']).reshape(-1,1))
#predictions = regression.predict(np.array(heights).reshape(-1,1))
#no_wingspan_data['Wingspan predictions (in cm)'] = predictions

```


```python
regression.score(np.array(heights).reshape(-1,1), np.array(wingspans))
```




    0.6885110373104122




```python
data=player_bio_1718[player_bio_1718['Wingspan (in cm)'] > 0]
x_dim = 'Height (in cm)'
y_dim = 'Wingspan (in cm)'
plt.scatter(
    x=data[x_dim],
    y=data[y_dim],
    label='Observations')
plt.scatter(
    x=data[x_dim],
    y=data['Wingspan predictions (in cm)'],
    label='Predictions from height')
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.legend()
plt.show()
```


    
![png](/images/NBA-bio/output_37_0.png)
    


This is not a great prediction, the $R^2$ score is $0.69$ out of $1.0$.

**The wingspan of the NBA population is more independant of the height of the players than the average population.**

#### Create BMI
Do NBA players have similar BMIs?
$$BMI = \frac{mass (kg)}{(height (m))^2}$$
Typically this measurement is not useful for atheletes because it does not distinguish fat weight from muscle weight. [Note](https://www.diabetes.ca/managing-my-diabetes/tools---resources/body-mass-index-(bmi)-calculator). 

We have height in cm and mass in pounds, so to convert it I will use the formula:
$$BMI = \frac{mass (lbs) / 2.2}{(height (cm)/100)^2}$$


```python
joined_data['BMI']= (joined_data['Weight (in lb)']/2.2)/((joined_data['Height (in cm)']/100)**2)
plot_metric_salary('BMI',joined_data)
```


    
![png](/images/NBA-bio/output_40_0.png)
    


We have some players coming in "overweight" according to BMI (BMI > 25), but, as mentioned above, it doesn't account for what type of tissue the weight is coming from. 

Actually, those guys are some of the higher paid players!

#### Create Hand Area
I think the surface area of the hand is more closely related to its impact on the game of basketball then either the width or length individually. I'm going to make that into a feature using: $$A = \pi * length * width$$


```python
joined_data['Hand Area (inches^2)'] = joined_data['Hand Width (in inches)']*joined_data['Hand Length (in inches)']
metric='Hand Area (inches^2)'
plot_metric_salary(metric,joined_data)
```


    
![png](/images/NBA-bio/output_43_0.png)
    


#### Create Age
I will use the birthdate column to calculate the age of the player in the 2017-2018 season. This season ended on May 31st, so I will take their age on June 1st. 


```python
joined_data['Birth Date'].head()
```




    0    1/14/1985
    1    9/16/1995
    2    9/25/1993
    3     6/3/1986
    4     1/4/1985
    Name: Birth Date, dtype: object




```python
joined_data['Birth Date']=pd.to_datetime(joined_data['Birth Date'])
joined_data['Age'] = 2018 - pd.DatetimeIndex(joined_data['Birth Date']).year
joined_data.Age.head()
```




    0    33
    1    23
    2    25
    3    32
    4    33
    Name: Age, dtype: int64




```python
metric='Age'
plot_metric_salary(metric,joined_data)
```


    
![png](/images/NBA-bio/output_47_0.png)
    


Who is in the 45+ category?


```python
joined_data[joined_data.Age > 45]
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
      <th>Player Full Name</th>
      <th>Birth Date</th>
      <th>Year Start</th>
      <th>Year End</th>
      <th>Position</th>
      <th>Height (ft 1/2)</th>
      <th>Height (inches 2/2)</th>
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>...</th>
      <th>College</th>
      <th>Player</th>
      <th>Unnamed: 0</th>
      <th>Tm</th>
      <th>season17_18</th>
      <th>Salary (Million USD)</th>
      <th>Log Salary (M $)</th>
      <th>BMI</th>
      <th>Hand Area (inches^2)</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>259</th>
      <td>Larry Nance</td>
      <td>1959-02-12</td>
      <td>2016</td>
      <td>2018</td>
      <td>F</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>205.7</td>
      <td>217.2</td>
      <td>274.3</td>
      <td>...</td>
      <td>University of Wyoming</td>
      <td>Larry Nance</td>
      <td>384</td>
      <td>CLE</td>
      <td>1471382.0</td>
      <td>1.471382</td>
      <td>0.167725</td>
      <td>24.707942</td>
      <td>87.75</td>
      <td>59</td>
    </tr>
    <tr>
      <th>389</th>
      <td>Tim Hardaway</td>
      <td>1966-09-01</td>
      <td>2014</td>
      <td>2018</td>
      <td>G</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>198.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>University of Michigan</td>
      <td>Tim Hardaway</td>
      <td>64</td>
      <td>NYK</td>
      <td>16500000.0</td>
      <td>16.500000</td>
      <td>1.217484</td>
      <td>23.744456</td>
      <td>NaN</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>



These are former pros with sons in the league with their same name. Per wikipedia, the biological stats for Larry Nance, and Tim Hardaway, match more closely to the **Juniors** (except for Birth Date), so I'm going to update the Birth Dates and recalculate the ages

https://en.wikipedia.org/wiki/Larry_Nance_Jr., https://en.wikipedia.org/wiki/Tim_Hardaway_Jr.,


```python
player_salary[player_salary.Player.str.contains('Nance')], player_salary[player_salary.Player.str.contains('Hardaway')]
```




    (     Unnamed: 0       Player   Tm  season17_18  Salary (Million USD)
     383         384  Larry Nance  CLE    1471382.0              1.471382,
         Unnamed: 0        Player   Tm  season17_18  Salary (Million USD)
     63          64  Tim Hardaway  NYK   16500000.0                  16.5)



These teams and salaries match the data from the 2017-2018 season.


```python
joined_data.loc[joined_data['Player']=='Larry Nance','Birth Date'] = '1993-01-01' #Larry Nance Jr.
joined_data.loc[joined_data['Player']=='Tim Hardaway','Birth Date'] = '1992-03-16' #Tim Hardaway Jr.
joined_data['Birth Date']=pd.to_datetime(joined_data['Birth Date'])
joined_data['Age'] = 2018 - pd.DatetimeIndex(joined_data['Birth Date']).year
```


```python
metric='Age'
plot_metric_salary(metric,joined_data)
```


    
![png](/images/NBA-bio/output_54_0.png)
    



```python
joined_data[joined_data.Age > 39]
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
      <th>Player Full Name</th>
      <th>Birth Date</th>
      <th>Year Start</th>
      <th>Year End</th>
      <th>Position</th>
      <th>Height (ft 1/2)</th>
      <th>Height (inches 2/2)</th>
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>...</th>
      <th>College</th>
      <th>Player</th>
      <th>Unnamed: 0</th>
      <th>Tm</th>
      <th>season17_18</th>
      <th>Salary (Million USD)</th>
      <th>Log Salary (M $)</th>
      <th>BMI</th>
      <th>Hand Area (inches^2)</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>Dirk Nowitzki</td>
      <td>1978-06-19</td>
      <td>1999</td>
      <td>2018</td>
      <td>F</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>213.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>Dirk Nowitzki</td>
      <td>201</td>
      <td>DAL</td>
      <td>5000000.0</td>
      <td>5.000000</td>
      <td>0.698970</td>
      <td>24.454263</td>
      <td>NaN</td>
      <td>40</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Jason Terry</td>
      <td>1977-09-15</td>
      <td>2000</td>
      <td>2018</td>
      <td>G</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>188.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>University of Arizona</td>
      <td>Jason Terry</td>
      <td>296</td>
      <td>MIL</td>
      <td>2328652.0</td>
      <td>2.328652</td>
      <td>0.367105</td>
      <td>23.792131</td>
      <td>NaN</td>
      <td>41</td>
    </tr>
    <tr>
      <th>274</th>
      <td>Manu Ginobili</td>
      <td>1977-07-28</td>
      <td>2003</td>
      <td>2018</td>
      <td>G</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>198.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>Manu Ginobili</td>
      <td>277</td>
      <td>SAS</td>
      <td>2500000.0</td>
      <td>2.500000</td>
      <td>0.397940</td>
      <td>23.744456</td>
      <td>NaN</td>
      <td>41</td>
    </tr>
    <tr>
      <th>416</th>
      <td>Vince Carter</td>
      <td>1977-01-26</td>
      <td>1999</td>
      <td>2018</td>
      <td>G-F</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>198.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>University of North Carolina</td>
      <td>Vince Carter</td>
      <td>143</td>
      <td>SAC</td>
      <td>8000000.0</td>
      <td>8.000000</td>
      <td>0.903090</td>
      <td>25.481856</td>
      <td>NaN</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 24 columns</p>
</div>



This checks out!

#### Create Years in the league


```python
joined_data['Years in the league'] = 2018 - joined_data['Year Start']
metric='Years in the league'
plot_metric_salary(metric,joined_data)
```


    
![png](/images/NBA-bio/output_58_0.png)
    


## Train the model
Before I do that, I definitely need to drop some columns. I'm going to leave categoricial data (College, Position, Team) in right now in case I want it later.


```python
joined_data.columns
```




    Index(['Player Full Name', 'Birth Date', 'Year Start', 'Year End', 'Position',
           'Height (ft 1/2)', 'Height (inches 2/2)', 'Height (in cm)',
           'Wingspan (in cm)', 'Standing Reach (in cm)', 'Hand Length (in inches)',
           'Hand Width (in inches)', 'Weight (in lb)', 'Body Fat (%)', 'College',
           'Player', 'Unnamed: 0', 'Tm', 'season17_18', 'Salary (Million USD)',
           'Log Salary (M $)', 'BMI', 'Hand Area (inches^2)', 'Age',
           'Years in the league'],
          dtype='object')




```python
dropped_columns = ['Player Full Name', 'Birth Date', 'Year Start', 'Year End',
       'Height (ft 1/2)', 'Height (inches 2/2)','Unnamed: 0','season17_18','Player']
categorical_columns = ['Tm','Position','College']
joined_data_dropped = joined_data.drop(columns=dropped_columns)
joined_data_dropped = joined_data_dropped.drop(columns=categorical_columns)
joined_data_dropped.head()
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
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>Hand Length (in inches)</th>
      <th>Hand Width (in inches)</th>
      <th>Weight (in lb)</th>
      <th>Body Fat (%)</th>
      <th>Salary (Million USD)</th>
      <th>Log Salary (M $)</th>
      <th>BMI</th>
      <th>Hand Area (inches^2)</th>
      <th>Age</th>
      <th>Years in the league</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>182.9</td>
      <td>193.0</td>
      <td>238.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
      <td>0.027</td>
      <td>2.116955</td>
      <td>0.325712</td>
      <td>21.876396</td>
      <td>NaN</td>
      <td>33</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>205.7</td>
      <td>212.7</td>
      <td>266.7</td>
      <td>8.75</td>
      <td>10.5</td>
      <td>220.0</td>
      <td>0.051</td>
      <td>5.504420</td>
      <td>0.740712</td>
      <td>23.633684</td>
      <td>91.875</td>
      <td>23</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>230.0</td>
      <td>NaN</td>
      <td>1.167333</td>
      <td>0.067195</td>
      <td>26.640122</td>
      <td>NaN</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>208.3</td>
      <td>215.3</td>
      <td>271.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>245.0</td>
      <td>0.091</td>
      <td>27.734405</td>
      <td>1.443019</td>
      <td>25.666394</td>
      <td>NaN</td>
      <td>32</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>208.3</td>
      <td>219.7</td>
      <td>279.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>289.0</td>
      <td>0.105</td>
      <td>9.769821</td>
      <td>0.989887</td>
      <td>30.275869</td>
      <td>NaN</td>
      <td>33</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



### Options for Nans
1. Work on dataset of only intact rows
1. Work on dataset of only intact columns
1. Replace nan with 
    - mean, 
    - mode, 
    - or median

#### Work on dataset with only intact rows
Let's start with 1, and see how they all compare


```python
training_data_intact_rows = joined_data_dropped.dropna()
#training_data_intact_rows=training_data_intact_rows.drop(columns=['Wingspan predictions (in cm)']) #because we are using real wingspan data
training_data_intact_rows.head()
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
      <th>Height (in cm)</th>
      <th>Wingspan (in cm)</th>
      <th>Standing Reach (in cm)</th>
      <th>Hand Length (in inches)</th>
      <th>Hand Width (in inches)</th>
      <th>Weight (in lb)</th>
      <th>Body Fat (%)</th>
      <th>Salary (Million USD)</th>
      <th>Log Salary (M $)</th>
      <th>BMI</th>
      <th>Hand Area (inches^2)</th>
      <th>Age</th>
      <th>Years in the league</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>205.7</td>
      <td>212.7</td>
      <td>266.7</td>
      <td>8.75</td>
      <td>10.50</td>
      <td>220.0</td>
      <td>0.051</td>
      <td>5.504420</td>
      <td>0.740712</td>
      <td>23.633684</td>
      <td>91.875</td>
      <td>23</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>198.1</td>
      <td>208.3</td>
      <td>262.9</td>
      <td>9.00</td>
      <td>8.25</td>
      <td>214.0</td>
      <td>0.051</td>
      <td>10.845506</td>
      <td>1.035250</td>
      <td>24.786896</td>
      <td>74.250</td>
      <td>27</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>215.9</td>
      <td>222.3</td>
      <td>0.0</td>
      <td>9.00</td>
      <td>10.75</td>
      <td>260.0</td>
      <td>0.064</td>
      <td>4.187599</td>
      <td>0.621965</td>
      <td>25.353936</td>
      <td>96.750</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>205.7</td>
      <td>221.6</td>
      <td>275.6</td>
      <td>9.50</td>
      <td>9.50</td>
      <td>220.0</td>
      <td>0.082</td>
      <td>7.319035</td>
      <td>0.864454</td>
      <td>23.633684</td>
      <td>90.250</td>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>198.1</td>
      <td>211.5</td>
      <td>262.9</td>
      <td>8.25</td>
      <td>8.50</td>
      <td>210.0</td>
      <td>0.047</td>
      <td>19.332500</td>
      <td>1.286288</td>
      <td>24.323589</td>
      <td>70.125</td>
      <td>26</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = training_data_intact_rows.drop(columns=['Salary (Million USD)','Log Salary (M $)'])
y = training_data_intact_rows[['Log Salary (M $)']]
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X,y)
print('score: '+str(model.score(X,y)))

```

    score: 0.6153114126697995
    


```python
training_data_filled = joined_data_dropped.fillna(joined_data_dropped.median())
X_filled = training_data_filled.drop(columns=['Salary (Million USD)','Log Salary (M $)'])
y_filled = training_data_filled[['Log Salary (M $)']]
model.score(X_filled,y_filled)
```




    -0.9542480606293884



This model is overfit - when I evaluate my dataset with other players given reasonable values for hand size, reach, wingspan, etc, it fails **spectacularly.** 


```python
y_filled.insert(0,"Salary (M $)",10**y_filled['Log Salary (M $)'])
y_filled.insert(0,"Predicted Log Salary", model.predict(X_filled))
y_filled.insert(0,"Predicted Salary (M $)",10**y_filled["Predicted Log Salary"])

```


```python
data=y_filled
x_dim = 'Salary (M $)'
y_dim = 'Salary (M $)'
plt.scatter(
    data=y_filled,
    x=x_dim,
    y=y_dim,
    label='Observations')
y_dim = 'Predicted Salary (M $)'
plt.scatter(
    data=y_filled,
    x=x_dim,
    y=y_dim,
    label='Predictions',
    alpha=0.5,)
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.legend()
plt.show()
```


    
![png](/images/NBA-bio/output_69_0.png)
    


Who is that we predict should be puling in $1B?


```python
joined_data.iloc[y_filled["Predicted Salary (M $)"].idxmax()]
```




    Player Full Name                 Dirk Nowitzki
    Birth Date                 1978-06-19 00:00:00
    Year Start                                1999
    Year End                                  2018
    Position                                     F
    Height (ft 1/2)                              7
    Height (inches 2/2)                          0
    Height (in cm)                           213.4
    Wingspan (in cm)                           NaN
    Standing Reach (in cm)                     NaN
    Hand Length (in inches)                    NaN
    Hand Width (in inches)                     NaN
    Weight (in lb)                             245
    Body Fat (%)                               NaN
    College                                    NaN
    Player                           Dirk Nowitzki
    Unnamed: 0                                 201
    Tm                                         DAL
    season17_18                              5e+06
    Salary (Million USD)                         5
    Log Salary (M $)                       0.69897
    BMI                                    24.4543
    Hand Area (inches^2)                       NaN
    Age                                         40
    Years in the league                         19
    Name: 105, dtype: object



Per Wikipeida, [Dirk Nowitzki](https://en.wikipedia.org/wiki/Dirk_Nowitzki) "is widely regarded as one of the greatest power forwards of all time and is considered by many to be the greatest European player of all time." 

So maybe the Mavs were getting a deal! **Or maybe the model is flawed.**

#### Work on dataset with only intact columns


```python
columns_with_na =[]
for column in joined_data_dropped.columns:
    #print(column, np.sum(joined_data_dropped[column].isna()))
    if np.sum(joined_data_dropped[column].isna()):
        columns_with_na.append(column)
```


```python
training_data_intact_cols=joined_data_dropped.drop(columns=columns_with_na)
training_data_intact_cols.head()
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
      <th>Height (in cm)</th>
      <th>Weight (in lb)</th>
      <th>Salary (Million USD)</th>
      <th>Log Salary (M $)</th>
      <th>BMI</th>
      <th>Age</th>
      <th>Years in the league</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>182.9</td>
      <td>161.0</td>
      <td>2.116955</td>
      <td>0.325712</td>
      <td>21.876396</td>
      <td>33</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>205.7</td>
      <td>220.0</td>
      <td>5.504420</td>
      <td>0.740712</td>
      <td>23.633684</td>
      <td>23</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198.1</td>
      <td>230.0</td>
      <td>1.167333</td>
      <td>0.067195</td>
      <td>26.640122</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>208.3</td>
      <td>245.0</td>
      <td>27.734405</td>
      <td>1.443019</td>
      <td>25.666394</td>
      <td>32</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>208.3</td>
      <td>289.0</td>
      <td>9.769821</td>
      <td>0.989887</td>
      <td>30.275869</td>
      <td>33</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = training_data_intact_cols.drop(columns=['Salary (Million USD)','Log Salary (M $)'])
y = training_data_intact_cols[['Log Salary (M $)']]
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X,y)
print('score: '+str(model.score(X,y)))
```

    score: 0.2710707378505274
    

The scores are lower for this - this is as expected. There are more examples and fewer categories. 

#### Work with filling NaN with median


```python
training_data_filled = joined_data_dropped.fillna(joined_data_dropped.median())
X = training_data_filled.drop(columns=['Salary (Million USD)','Log Salary (M $)'])
y = training_data_filled[['Log Salary (M $)']]
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X,y)
print('score: '+str(model.score(X,y)))
```

    score: 0.2827808374769417
    

## Test the model

### Test/train split

### Preprocessing
I'm going to try some polynominal features here on the "intact columns" dataset. 

I'm choosing that because many of the features I added are multiplicative of one another, so adding polynominal features will only amplify this. 

I'm also choosing to drop BMI for the same reason: it's a product of height and weight.


```python
X = training_data_intact_cols.drop(columns=['Salary (Million USD)','Log Salary (M $)','BMI'])
y = training_data_intact_cols[['Log Salary (M $)']]
```


```python
model = make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), LinearRegression())
model.fit(X,y)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
np.mean(cross_val_score(model, X, y, cv=cv))
```




    0.42616838460075873




```python
model = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression())
model.fit(X,y)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
np.mean(cross_val_score(model, X, y, cv=cv))
```




    0.4519833642191813



Becuase the test/train split scores are lower with polynominal degree of 3, I conclude that is causing overfitting. I know that 0.45 is a pretty medicore $R^2$ score, but it's the best I could do. Let's see what my salary would be!

### Predicitions


```python
model.feature_names_in_
```




    array(['Height (in cm)', 'Weight (in lb)', 'Age', 'Years in the league'],
          dtype=object)




```python
RWL_stats = {
            X.columns[0]:[180.4],  #5'11" (height in cm)
            X.columns[1]:[175], # weight in lbs
            X.columns[2]:[29], #age
            X.columns[3]:[0], #years in league
            }
RWL_df = pd.DataFrame.from_dict(RWL_stats)
RWL_df
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
      <th>Height (in cm)</th>
      <th>Weight (in lb)</th>
      <th>Age</th>
      <th>Years in the league</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>180.4</td>
      <td>175</td>
      <td>29</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
log10_RWL_salary = model.predict(RWL_df)
RWL_salary = 10**log10_RWL_salary
RWL_salary[0][0]
```




    1.213584732487101



So, a fair salary for me would be $1,200,000 a year. 

Coach Ham, I already live in LA. My application is in the mail!



```python
y_pred = y
y_pred.insert(0,"Salary (M $)",10**y_pred['Log Salary (M $)'])
y_pred.insert(0,"Predicted Log Salary", model.predict(X))
y_pred.insert(0,"Predicted Salary (M $)",10**y_pred["Predicted Log Salary"])
```


```python
x_dim = 'Salary (M $)'
y_dim = 'Salary (M $)'
plt.scatter(
    data=y_pred,
    x=x_dim,
    y=y_dim,
    label='Observations')
y_dim = 'Predicted Salary (M $)'
plt.scatter(
    data=y_pred,
    x=x_dim,
    y=y_dim,
    label='Predictions',
    alpha=0.5,)
plt.scatter(
    x=0,
    y=RWL_salary,
    label='RWL',
    alpha=1,)
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.legend()
plt.show()
```


    
![png](/images/NBA-bio/output_93_0.png)
    



```python
plt.subplots(1,2,figsize=(22,7))
plt.subplot(1,2,1)
data=y_filled
x_dim = 'Salary (M $)'
y_dim = 'Salary (M $)'
plt.scatter(
    data=y_filled,
    x=x_dim,
    y=y_dim,
    label='Observations')
y_dim = 'Predicted Salary (M $)'
plt.scatter(
    data=y_filled,
    x=x_dim,
    y=y_dim,
    label='Predictions',
    alpha=0.5,)
plt.xlabel(x_dim)
plt.ylabel(y_dim)
#plt.fig_size=(11,7)
plt.title('Predictions with all biometrics')
plt.ylim([0,1100])
plt.legend()
plt.subplot(1,2,2)
x_dim = 'Salary (M $)'
y_dim = 'Salary (M $)'
plt.scatter(
    data=y_pred,
    x=x_dim,
    y=y_dim,
    label='Observations')
y_dim = 'Predicted Salary (M $)'
plt.scatter(
    data=y_pred,
    x=x_dim,
    y=y_dim,
    label='Predictions',
    alpha=0.5,)
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.ylim([0,1100])
plt.title('Predictions with height, weight, age, and years')
#plt.fig_size=(11,7)
plt.legend()
plt.show()
```


    
![png](/images/NBA-bio/output_94_0.png)
    


When viewed on the same scale, the model with fewer features is doing much better. But it's $R^2$ score was still only 0.45, which is not great. We must wonder:

## Why is this model doing so badly?
Or, why am worth more than minimum wage for an NBA player?


```python
coef_df = pd.DataFrame(
    zip(
    list(model['polynomialfeatures'].get_feature_names_out(model.feature_names_in_)),
    list(model['linearregression'].coef_,)[0]), columns=['Category','Coefficient']
)
coef_df.insert(2,'Magntitute of coef',np.abs(model['linearregression'].coef_,)[0])
coef_df.sort_values(by=['Magntitute of coef'],ascending=False)
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
      <th>Category</th>
      <th>Coefficient</th>
      <th>Magntitute of coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Years in the league</td>
      <td>0.611955</td>
      <td>0.611955</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Age Years in the league</td>
      <td>-0.480396</td>
      <td>0.480396</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Age</td>
      <td>-0.274165</td>
      <td>0.274165</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Age^2</td>
      <td>0.184503</td>
      <td>0.184503</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Years in the league^2</td>
      <td>0.118761</td>
      <td>0.118761</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Height (in cm)</td>
      <td>0.046181</td>
      <td>0.046181</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Height (in cm) Age</td>
      <td>0.042721</td>
      <td>0.042721</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Height (in cm) Years in the league</td>
      <td>-0.031671</td>
      <td>0.031671</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Height (in cm) Weight (in lb)</td>
      <td>0.028091</td>
      <td>0.028091</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Weight (in lb) Age</td>
      <td>0.023829</td>
      <td>0.023829</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Height (in cm)^2</td>
      <td>-0.020076</td>
      <td>0.020076</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Weight (in lb) Years in the league</td>
      <td>-0.019643</td>
      <td>0.019643</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Weight (in lb)</td>
      <td>-0.017419</td>
      <td>0.017419</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Weight (in lb)^2</td>
      <td>-0.005985</td>
      <td>0.005985</td>
    </tr>
  </tbody>
</table>
</div>



I really like this analysis becuase it finds that **age and years in the league are more important than height or weight.** 

Not only that, but the quatratic features of $Age^2$ and $years^2$ match my intuition and represent two different cases of high salary:

Young players have high potential: teams are eager to get young, talented players, and incentivize them with high salaries.  

Players with more years in the league are more experienced and known quantities, which is valuable in a different way.

$Age^2$ has a positive coeficient with high magntidue, because there players at **both** ends of that parabola are valuable. I draw a similar conclusion from the high magnitude, positive coefficeint on $years^2$


```python
X_uniform = pd.DataFrame({"Age":np.linspace(20,40,num=len(X)), "Years in the league":np.linspace(0,20,num=len(X))})
X.update(X_uniform)
scaled_X=model['standardscaler'].transform(X)
scaled_age = scaled_X[:,2]
scaled_years=scaled_X[:,3]
age_salary = scaled_age*(-0.274)+(scaled_age**2)*(0.1845)
years_salary =scaled_years*(0.6119)+(scaled_years**2)*(0.11876)

```


```python
plt.subplots(1,2,figsize=(22,7))
plt.subplot(1,2,1)
x_dim='Age'
y_dim='Salary (Million USD)'
plt.scatter(
    data=joined_data,
    x=x_dim,
    y=y_dim,
    label='Observations',
)
plt.plot(X_uniform['Age'],10**age_salary,
        label="Contribution from age only",
         color='#ff7f0e')
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.ylim([-2,38])
plt.legend()

plt.subplot(1,2,2)
x_dim='Years in the league'
plt.scatter(
    data=joined_data,
    x=x_dim,
    y=y_dim,
    label='Observations',
)
plt.plot(X_uniform['Years in the league'],10**years_salary,
        label="Contribution from years only",
         color='#ff7f0e')
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.ylim([-2,38])
plt.legend()
plt.show()
```


    
![png](/images/NBA-bio/output_100_0.png)
    


This calcuation neglected the cross term of $age*years$. Let's include it and re-evaluate.


```python
X_uniform = pd.DataFrame({"Age":np.linspace(20,40,num=len(X)), "Years in the league":np.linspace(0,20,num=len(X))})
X.update(X_uniform)
scaled_X=model['standardscaler'].transform(X)
scaled_age = scaled_X[:,2]
scaled_years=scaled_X[:,3]
age_years_salary = scaled_age*(-0.274)+(scaled_age**2)*(0.1845)+(scaled_age*scaled_years)*(-0.4804)
x_dim='Age'
y_dim='Salary (Million USD)'
plt.scatter(
    data=joined_data,
    x=x_dim,
    y=y_dim,
    label='Observations',
)
plt.plot(X_uniform['Age'],10**age_years_salary,
        label="Contribution from age and years",
         color='#ff7f0e')
plt.xlabel(x_dim)
plt.ylabel(y_dim)
plt.ylim([-2,38])
plt.legend()
plt.show()
```


    
![png](/images/NBA-bio/output_102_0.png)
    


The little bump centered around 23 years old is showing that NBA salaries are rewarding the rare combination of low age and high experience. After an age of about 28, the negatives of age overtakes benefit from experience.
