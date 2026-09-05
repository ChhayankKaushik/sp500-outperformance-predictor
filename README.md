# sp500-outperformance-predictor
A custom Random Forest classifier built from scratch in Python to predict S&amp;P 500 stock outperformance using quarterly fundamental and market technical metrics. A custom DecisionTreeClassifier and RandomForestClassifier classes built without scikit-learn to demonstrate an understanding of the tree building algorithms, bootstrap ensembling, gini impurity and information gain. The training window (2010–2019) and out-of-sample testing window (2021–2024) are separated to prevent lookahead bias, making sure the testing window years are after the training window years. The market technical metrics used were: 12-month Trailing Momentum, Annual Rolling Volatility, yearly and quarterly Maximum Drawdown, Proximity to 52 week High, Average Trading Volume per day over year, RVOL, Log Dollar Volume and Beta vs S&P 500. These specific metrics were used as it helped get signals from factors such as momentum, volatility, risk and liquidity.
To analyse the performance of the algorithm precision, recall, specificity, F1-score are calculated. One run of the algorithm gives these results:

--- Classification Diagnostics ---
True Positives (Correct BUYs):   168
False Positives (Failed BUYs):   195
True Negatives (Correct AVOIDs): 718
False Negatives (Missed WINs):  519

--- Performance Scores ---
Precision (BUY Win Rate): 46.28%
Recall (Winners Caught):  24.45%
F1-Score:                 0.3200
Specificity (Avoidance):  78.64%

It is clear that the model is a lot better at 
