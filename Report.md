### Intro

The goal of this project is to test whether future stock market trends can be predicted based off of early trends. Our plan is to implement and compare a variety of data mining models to approach this problem. This is a traditionally difficult problem since the stock market is highly dependent on a number of difficult to quantify factors. Public perception and product announcements can cause temporary swings in the price before it returns to a more normal value. The price is also affected by the actions of individuals, which can often act seemingly irrationally. 

### Methods

To implement naive-bayes clustering, the data was preprocessed to associate the previous n days of market data with the current day. The number of days to look back was variable to allow for many iterations of the model to be tested. By varying the number of days, the model could be adjusted and fine tuned as needed to optimize between speed, accuracy, and overfitting. The input data to the model was the difference in the opening and closing value of the market for a given day. This allowed each day to be easily summarised by a float valued gain or loss for the day, providing an easy metric to classify by. Training and classification were performed by the Naive-bayes clustering model from the sklearn library, both the Gaussian and Bernoulli versions of the model were used.

The models were also trained and tested on a variety of stock market tickers to ensure they were generally applicable. Random sampling and validation sets were used to verify that the models were working and not overfitting to a specific set of data. 

### Code

Naive-bayes and preprocessing: https://colab.research.google.com/drive/12mX8zQNEV5UAgeutjcD05HoH8n50_ONR
