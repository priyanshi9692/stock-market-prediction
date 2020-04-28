```python
# The goal of this project is to understand stock dataset of a particular company and then analyse
# the dataset to predict the future price. The prediction of stock market is very cumbersome process. 
# So, in this work I am implementing Support Vector Regression method. I am using three models,
# Linear Regression, Radical Basic Function and Polynomial Regression models.
```


```python
# Importing dependencies
import requests
import json
import csv
import numpy as np
import pandas as pd
```


```python
#First of all I am fetching Data from Alpha Vantage API,
# I am fetching Stock data for Adobe since last 20years to till date.
```


```python
# Collecting stock data of Adobe using Alpha Vantage API
CSV_URL ="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ADBE&apikey=NR8HXOOB06DUO2M3&datatype=csv&outputsize=full"
with requests.Session() as s:
    download = s.get(CSV_URL)
    print(download)
    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    df = pd.DataFrame(cr)
df
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header
df
```

    <Response [200]>





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
      <th>timestamp</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-04-27</td>
      <td>348.8000</td>
      <td>349.9600</td>
      <td>345.1200</td>
      <td>348.5000</td>
      <td>2361714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-24</td>
      <td>338.0000</td>
      <td>344.7000</td>
      <td>333.5011</td>
      <td>344.1000</td>
      <td>2146162</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-23</td>
      <td>338.9000</td>
      <td>341.7500</td>
      <td>334.8000</td>
      <td>335.3700</td>
      <td>2120251</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-22</td>
      <td>337.4100</td>
      <td>339.3800</td>
      <td>332.8006</td>
      <td>336.7700</td>
      <td>2231819</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-04-21</td>
      <td>340.9000</td>
      <td>342.7630</td>
      <td>325.2100</td>
      <td>328.9900</td>
      <td>3548202</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5028</th>
      <td>2000-05-02</td>
      <td>128.6000</td>
      <td>128.8000</td>
      <td>115.3000</td>
      <td>115.5000</td>
      <td>1736700</td>
    </tr>
    <tr>
      <th>5029</th>
      <td>2000-05-01</td>
      <td>122.2000</td>
      <td>131.0000</td>
      <td>122.1000</td>
      <td>130.7000</td>
      <td>2012300</td>
    </tr>
    <tr>
      <th>5030</th>
      <td>2000-04-28</td>
      <td>124.8000</td>
      <td>127.9000</td>
      <td>118.0000</td>
      <td>120.9000</td>
      <td>2832200</td>
    </tr>
    <tr>
      <th>5031</th>
      <td>2000-04-27</td>
      <td>108.1000</td>
      <td>124.5000</td>
      <td>107.0000</td>
      <td>123.4000</td>
      <td>2784800</td>
    </tr>
    <tr>
      <th>5032</th>
      <td>2000-04-26</td>
      <td>118.6000</td>
      <td>120.8000</td>
      <td>112.3000</td>
      <td>114.9000</td>
      <td>2502200</td>
    </tr>
  </tbody>
</table>
<p>5032 rows × 6 columns</p>
</div>




```python
# Once the data is collected I am applying Data Preprocession methods to clean data.
# Data Type of the columns
dataTypeOfColumns = df.dtypes
print(dataTypeOfColumns)
```

    0
    timestamp    object
    open         object
    high         object
    low          object
    close        object
    volume       object
    dtype: object



```python
# Checking if columns are unique column
columns = list(df)
for i in columns:
    print(i , "-", df[i].nunique() == df[i].count())
```

    timestamp - True
    open - False
    high - False
    low - False
    close - False
    volume - False



```python
# Finding if there is any NaN value
df.isnull().values.any()
```




    False




```python
# Updating Datatypes of the columns
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.open=df.open.astype(float)
df.high=df.high.astype(float)
df.low=df.low.astype(float)
df.close=df.close.astype(float)
df.volume=df.volume.astype(int)
df.dtypes
```




    0
    timestamp    datetime64[ns]
    open                float64
    high                float64
    low                 float64
    close               float64
    volume                int64
    dtype: object




```python
# By taking average of the high and low value I am calculating a mid or 
# average value of the stock for a particular day
df['average_value'] = df[['high', 'low']].mean(axis=1)
df
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
      <th>timestamp</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>average_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-04-27</td>
      <td>348.80</td>
      <td>349.960</td>
      <td>345.1200</td>
      <td>348.50</td>
      <td>2361714</td>
      <td>347.54000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-24</td>
      <td>338.00</td>
      <td>344.700</td>
      <td>333.5011</td>
      <td>344.10</td>
      <td>2146162</td>
      <td>339.10055</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-23</td>
      <td>338.90</td>
      <td>341.750</td>
      <td>334.8000</td>
      <td>335.37</td>
      <td>2120251</td>
      <td>338.27500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-22</td>
      <td>337.41</td>
      <td>339.380</td>
      <td>332.8006</td>
      <td>336.77</td>
      <td>2231819</td>
      <td>336.09030</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-04-21</td>
      <td>340.90</td>
      <td>342.763</td>
      <td>325.2100</td>
      <td>328.99</td>
      <td>3548202</td>
      <td>333.98650</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5028</th>
      <td>2000-05-02</td>
      <td>128.60</td>
      <td>128.800</td>
      <td>115.3000</td>
      <td>115.50</td>
      <td>1736700</td>
      <td>122.05000</td>
    </tr>
    <tr>
      <th>5029</th>
      <td>2000-05-01</td>
      <td>122.20</td>
      <td>131.000</td>
      <td>122.1000</td>
      <td>130.70</td>
      <td>2012300</td>
      <td>126.55000</td>
    </tr>
    <tr>
      <th>5030</th>
      <td>2000-04-28</td>
      <td>124.80</td>
      <td>127.900</td>
      <td>118.0000</td>
      <td>120.90</td>
      <td>2832200</td>
      <td>122.95000</td>
    </tr>
    <tr>
      <th>5031</th>
      <td>2000-04-27</td>
      <td>108.10</td>
      <td>124.500</td>
      <td>107.0000</td>
      <td>123.40</td>
      <td>2784800</td>
      <td>115.75000</td>
    </tr>
    <tr>
      <th>5032</th>
      <td>2000-04-26</td>
      <td>118.60</td>
      <td>120.800</td>
      <td>112.3000</td>
      <td>114.90</td>
      <td>2502200</td>
      <td>116.55000</td>
    </tr>
  </tbody>
</table>
<p>5032 rows × 7 columns</p>
</div>




```python
# Rearranging Columns
column_titles = ['timestamp','open','high','average_value','low','close','volume']
df.reindex(columns = column_titles)
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
      <th>timestamp</th>
      <th>open</th>
      <th>high</th>
      <th>average_value</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-04-27</td>
      <td>348.80</td>
      <td>349.960</td>
      <td>347.54000</td>
      <td>345.1200</td>
      <td>348.50</td>
      <td>2361714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-24</td>
      <td>338.00</td>
      <td>344.700</td>
      <td>339.10055</td>
      <td>333.5011</td>
      <td>344.10</td>
      <td>2146162</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-23</td>
      <td>338.90</td>
      <td>341.750</td>
      <td>338.27500</td>
      <td>334.8000</td>
      <td>335.37</td>
      <td>2120251</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-22</td>
      <td>337.41</td>
      <td>339.380</td>
      <td>336.09030</td>
      <td>332.8006</td>
      <td>336.77</td>
      <td>2231819</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-04-21</td>
      <td>340.90</td>
      <td>342.763</td>
      <td>333.98650</td>
      <td>325.2100</td>
      <td>328.99</td>
      <td>3548202</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5028</th>
      <td>2000-05-02</td>
      <td>128.60</td>
      <td>128.800</td>
      <td>122.05000</td>
      <td>115.3000</td>
      <td>115.50</td>
      <td>1736700</td>
    </tr>
    <tr>
      <th>5029</th>
      <td>2000-05-01</td>
      <td>122.20</td>
      <td>131.000</td>
      <td>126.55000</td>
      <td>122.1000</td>
      <td>130.70</td>
      <td>2012300</td>
    </tr>
    <tr>
      <th>5030</th>
      <td>2000-04-28</td>
      <td>124.80</td>
      <td>127.900</td>
      <td>122.95000</td>
      <td>118.0000</td>
      <td>120.90</td>
      <td>2832200</td>
    </tr>
    <tr>
      <th>5031</th>
      <td>2000-04-27</td>
      <td>108.10</td>
      <td>124.500</td>
      <td>115.75000</td>
      <td>107.0000</td>
      <td>123.40</td>
      <td>2784800</td>
    </tr>
    <tr>
      <th>5032</th>
      <td>2000-04-26</td>
      <td>118.60</td>
      <td>120.800</td>
      <td>116.55000</td>
      <td>112.3000</td>
      <td>114.90</td>
      <td>2502200</td>
    </tr>
  </tbody>
</table>
<p>5032 rows × 7 columns</p>
</div>




```python
# Plotting curve TimeStamp v/s Volume
import matplotlib.pyplot as plt

df_updated = df.head(20)
df_updated.plot(kind='line',x = "timestamp",y = "volume")
plt.title('Timestamp v/s Volume curve')
plt.xlabel('TIMESTAMP')
plt.ylabel('STOCK VOLUME')
plt.show()
```


![png](output_10_0.png)



```python
# Plotting Bar Graph between TimeStamp and Close Value
import matplotlib.pyplot as plt

df_updated = df.head(30)

df_updated.plot(kind='bar',x = "timestamp",y = "close")
plt.title('Timestamp v/s close value')
plt.xlabel('TIMESTAMP')
plt.ylabel('CLOSE VALUE')
plt.show()

```


![png](output_11_0.png)



```python
# Plotting curve TimeStamp v/s Close Value
# Here I am updating df_updated by collecting data of the recent 14 days.
df_updated = df.head(14)
df_updated.plot(kind='line',x = "timestamp",y = "close")
plt.title('Timestamp v/s close')
plt.xlabel('TIMESTAMP')
plt.ylabel('CLOSE')
plt.show()
```


![png](output_12_0.png)



```python
# importing dependencies
from sklearn.svm import SVR 
%matplotlib inline
```


```python
# Get data function
def get_data(df):  
    df.timestamp = df.timestamp.astype(str)
    data = df.copy()
    data['timestamp'] = data['timestamp'].str.split('-').str[2]
    data['timestamp'] = pd.to_numeric(data['timestamp'])
    return [ data['timestamp'].tolist(), data['close'].tolist() ] 
# Convert Series to list
dates, prices = get_data(df_updated)
```

    /usr/local/lib/python3.7/site-packages/pandas/core/generic.py:5303: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[name] = value



```python
# Use of three models: linear, polynomial and radial basic function (default kernal for SVR)
def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # convert to 1xn dimension
    x = np.reshape(x,(len(x), 1))
    
    svr_lin  = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=4)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.5)
    
    # Fit regression model
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
```


```python
# Predicted Price for the date 27th April
predicted_price = predict_prices(dates, prices, [27])
print(predicted_price)
```


![png](output_16_0.png)


    (348.4001955092228, 348.60000000012303, 349.9899665829195)



```python
predicted_price = list(predicted_price)
print(predicted_price)
```

    [348.4001955092228, 348.60000000012303, 349.9899665829195]



```python
# Calculating Root Mean Square Error
from sklearn.metrics import mean_squared_error
from math import sqrt
actual_value = df['close'].at[1]
type(actual_value)
actual_value = actual_value.item()
# Actual_Value fetching from the dataframe
print('Actual Value: ', actual_value)
# Predicted_Value from Linear Regression model
predicted_value_RBFRegression = predicted_price[0]
predicted_value_linearRegression = predicted_price[1]
predicted_value_polynomial = predicted_price[2]
print('Predicted value from Radical Basic Model: ', predicted_value_RBFRegression)
print('Predicted value from Linear Regression Model: ', predicted_value_linearRegression)
print('Predicted value from Polynomial Model: ', predicted_value_polynomial)
actual_value1 = []
actual_value1.append(actual_value)
predicted_value1 = []
predicted_value1.append(predicted_value_linearRegression)
# Mean Square Error
mse = mean_squared_error(actual_value1, predicted_value1)
rmse = sqrt(mse)
print("Root Mean Square Error value: ", rmse)
```

    Actual Value:  348.5
    Predicted value from Radical Basic Model:  348.4001955092228
    Predicted value from Linear Regression Model:  348.60000000012303
    Predicted value from Polynomial Model:  349.9899665829195
    Root Mean Square Error value:  0.1000000001230319



```python
# The results shows that all the three models provides very close results to the actual value,
# when we do analysis on the last 15days data, 
# This proves Support Vector Regression method models provides very good accuracy and least root mean square values.
```


```python
# References: 
# https://itnext.io/learning-data-science-predict-stock-price-with-support-vector-regression-svr-2c4fdc36662
# https://towardsdatascience.com/walking-through-support-vector-regression-and-lstms-with-stock-price-prediction-45e11b620650
```


```python

```
