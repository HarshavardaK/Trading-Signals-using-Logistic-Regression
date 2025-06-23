# Trading-Signals-using-Logistic-Regression
# Trading Signals using Logistic Regression

### ~By Harshavarda Kumarasamy (23NA10017)

---

## Task

To create a basic framework for algorithmic trading that can predict price direction using  
Multivariate Logistic Regression to assign ‘Buy’, ’Sell’, ’Hold’ positions on a one-day ticker  
basis such that the algorithm is profitable and accurate.

---

## Objective

Build an algorithm such that the algorithm is profitable and accurate.

---

## Data

I have used TCS NIFTY data and have opted for the 2013 to 2018 data to train the model.  
I have also further back tested the model’s efficiency on 2006-2007 data, 2020-2022 data,  
2018-2024 data and December 2023-March 2024 data of the same company to observe  
profits.

**I have selected TCS because:**

- It is a well-reputed company with relatively stable price variation in the stock market.
- The data set from 2013 to 2018 is relatively flat with brief uptrends. This provides an abundance of data for training of the model.
- The dip in stock prices during COVID is quite low compared to some other companies, making it ideal for using both pre-COVID and post-COVID data both.

**Why I selected these particular timeframes for back testing:**

- 2006 to 2007 show a relatively flat market similar to the dataset that was used for training. It also shows the performance of model in old market.
- 2020 to 2022 for the most part shows a rapid uptrend. This is to test if the model can function efficiently in a different trend than what it was trained in.
- 2018-2024 shows a model that has multiple trend changes. It is flat at first, followed by a brief downtrend, then an uptrend, then flat again.
- December 2023-March 2024 shows whether this model is viable in current market.

---

## Order of Code

1. Defining Regression function  
2. Defining Scaling, Accuracy, F1 score, AUC-ROC functions  
3. Defining Technical Indicator functions  
4. Defining Feature Engineering function  
5. Defining Plotting functions  
6. Importing data from yfinance and labelling it  
7. Defining Target and Features  
8. Splitting the dataset into training data and testing data  
9. Training the model  
10. Applying model on the testing data  
11. Model Evaluation of overall dataset (Accuracy, F1 score, Percent Profit/Loss)  
12. Back Testing  

---

## Strategy

I have initially calculated Price Direction of the stock market on a daily basis. If the market is  
to go up the next day, 1 is stored and 0 for when the market goes down the next day. I have  
used this initial calculation as a label for the prediction of the required target variable.

If day label is 1 and there have been no shares bought previously, take a Buy position at  
the next day’s closing price. Taking an initial sum of around 12,000 INR. When buy position is  
hit then the algorithm buys as many shares as possible. When label turns to 0, then sell  
position is taken. All shares of stock are sold. For the remaining days ‘hold’ position is  
maintained.

**Why 12,000 INR?**

As you can see from the image above, the current price of TCS is around 3900 INR. Taking  
multiples of 4000 INR would be efficient as even starting with high stock price, it will still be  
possible for the algorithm to buy multiple shares. I had initially attempted 4000 INR,  
8000 INR, 12000 INR and found among the three 12000 INR showed more profitability,  
although there wasn’t much difference amongst the three.

---

## Explaining `MultiClassLogisticRegression()` function

As there are three target variables, I have used Multiclass variation. I have implemented  
some basic principles of Softmax regression, One-vs-All strategy (Converting all labels to  
Binary Classifiers), K-fold splits and Regularisation in this code.

The One-vs-All is dependent on building binary classifiers (three in this case). These binary  
classifiers look at the data and check if one particular category is present (1 is assigned). If  
not, 0 is assigned. (For example, let us look at binary classifier for Buy — if `price_data[‘target’]`  
is 2 for a particular row, it is labelled as 1. Otherwise (target = 1 or 0), it is labelled 0.)

I have also defined a Binary Logistic Regression function with K-fold splits and Regularisation.  
K-fold split divides the data sets into a specified number of parts for more efficient  
training. This function operates on the binary classifiers to find the probability of one  
particular category (Buy, Sell or Hold).

This code predicts the Softmax value for each set category and chooses the one that shows  
maximum probability.

This forms something similar to the output layer of Softmax regression neural network.

I have divided the training set into 9 sets as this showed most efficiency during manual  
testing. I’ve set the code to run for 2500 iterations because despite the long run time, the  
model shall only undergo training phase once throughout the whole code. I felt it was better  
to create a more accurate model.

---

## Indicators Used and Why

### Following the Trend

- **SMA, EMA**: Smooth out price fluctuations and identify overall trend (upward, downward, or flat).
- **MACD**: Pinpoint potential trend reversals based on the difference between two moving averages.

### Measuring Volatility and Market Sentiment

- **Bollinger Bands**:  
  Create a channel around the moving average, highlighting price volatility.  
  - Prices exceeding the upper band → overbought  
  - Prices breaching the lower band → oversold

- **Average True Range (ATR)**:  
  Measures the typical range of price movement over a period.  
  - Higher ATR → more volatile market.

### Identifying Turning Points and Momentum

- **Stochastic Oscillator and RSI**:  
  Compare closing price to recent range, giving overbought/oversold signals.  
  - High reading → possible pullback  
  - Low reading → possible rebound

- **On-Balance Volume (OBV)**:  
  Accumulates volume based on price movement to indicate buying/selling pressure.  
  - Rising OBV with rising prices → strong buying  
  - Falling OBV with rising prices → weakening interest

---

## Feature Engineering

I create new features from existing data to understand price movements better.

### Two Key Examples:

1. **Bollinger Band Width**
   - Captures volatility. Wider bands → higher volatility.
   - Common strategy: increase in width → trend forming; constriction → flat market.

2. **SMA/RSI Ratio**
   - Combines trend (SMA) and momentum (RSI).
   - High ratio → continuing uptrend despite overbought  
   - Low ratio → potential trend reversal

These features help capture relationships between price, volatility, and momentum for better predictions.

---

## Why I Didn’t Implement Support and Resistance

- Though it could increase profits, S&R may not generalize well across datasets with different volatility.
- The model is trained on a flat market and works on short-term direction shifts — akin to scalping.
- In scalping, fixed thresholds like 1.10x are hard to define and apply universally.
- Goal: Generalize model across different market types; S&R would need tuning for each.

---

## Percent Profit/Loss Metric Explained

In addition to Accuracy, F1 score, and AUC-ROC, I use a **Percent Profit** metric.  

- Starting with ₹12,000 investment.
- Evaluates how much capital grows/losses using predicted values.
- Fits well with the logic of our algorithm and reflects its **real trading performance**.

---

## Overall Model Performance

- Backtests show consistent, sizeable profits across all four test periods.
- Accuracy, F1 score, and AUC-ROC are close to training performance.
- Indicates **strong generalization** capability.

---

## Some Potential Shortcomings

- Slightly **low accuracy** due to:
  - Labels based on price action  
  - Technical indicators tend to **lag** behind real price movement
