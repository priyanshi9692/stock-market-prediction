# Stock Prediction using Machine Learning Algorithms
The stock market is one of the most complicated and lucrative businesses. Predicting  stock  trends  is  a  cumbersome task,  as  the  stock  market  is  very  dynamic  and  composedof  various  factors.  The  main  factors  involved  are  Physical, Psychological, Rational, Irrational behavior, etc. Here, we have applied four ML algorithms to predict stock price. 
1. Neural Network computation framework
2. Time series analysis using ARIMA
3. Support Vector Regression models: Linear Regression, Polynomial and Radical Basic Function models
4. Naive-Bayes clustering model
## Environment
We use **Jupyter Notebook** to implement the mentioned models on Stock Market Data. 

## Importing Dependencies
We imported some modules prior to the implementation, those are,
1. **Pandas** python library,
2. **NumPy** python library,
3. **Matplotlib** python library to plot curves and
4. **Requests** has been installed to GET data from Alpha Vantage API or Yahoo Finance API.
5. **SKlearn libraries** for the implementation of models etc.

## Getting the Data
The data for all the implementations is fetched directly from online sources like, *Alpha Vantage APIs*, *Yahoo Finance*, etc. Jupyter Notebook is used to fetch data directly from these sources. 

## Running the Models
The project is built in components. Every component has implementation code for Stock data Analysis. So each component can be run individually to test a different model. All the models were built and run on jupyter notebooks, so they can be run by downloading and running the jupyter notebooks.
1. **Akshata Deo** implemented Neural Network Analysis. [Neural Network Analysis](https://github.com/priyanshi9692/stock-market-prediction/blob/master/NeuralNet_using_TF.ipynb)
![Neural Network Analysis: Actual vs Predicted Values](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Visualizations/NN_actualVSpredicted.png)

2. **Charlie Brayton** implemented Naive-Bayes model. [Naive-Bayes Model](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Naive_Bayes_Stock_Prediction.ipynb)
![Naive-Bayes Model: Histogram](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Visualizations/Naive-Bayes%20histogram.png)

3. **Dikshita Borkakati** implemented ARIMA model. [ARIMA Model](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Stock_Prediction_ARIMA_final.ipynb)
![ARIMA model: Actual vs Predicted Values](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Visualizations/Actual_vs_predicted.PNG)

4. **Priyanshi Jajoo** implemented SVR models, **Linear Regression, Polynomial and Radical Basic Function**. [SVR Model](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Stock-Price-Prediction-Using-Support-Vector-Regression-Method/Stock-Price-Prediction-Using-Support-Vector-Regression-Method.ipynb)
![Support Vector Regression Curve: Linear Regression, Polynomial, and Radical Basic Function](https://github.com/priyanshi9692/stock-market-prediction/blob/master/Visualizations/output_18_0.png)

## Comparing Models
Overall the stock predictions performed surprisingly well. The Neural Network prediction especially performed incredibly well with a **MSE of 0.00012404985**. Given this level of accuracy, future extensions of the project could test this model with stocks from other companies to help further validate the model and ensure it isn't over fitting.
The Support Vector Regression model also performed exceedingly well, predicting the stock price to within a dollar based off of the prior 15 days.
Based off of these models and the ARIMA model, which also produced quality results with a MSE of 3.694, we were able to successfully and accurately predict stock values based on prior days stock events. 
The success of these models helps explain the success and rise of algo-trading, which is currently used to make numerous small profitable trades in the stock market. The primary difference between our models and algo-trader's is that our models used daily data, while algo-traders use a live stream of current data.
