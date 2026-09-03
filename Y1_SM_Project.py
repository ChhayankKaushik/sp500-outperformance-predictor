# ==========================================
# IMPORTS
# ==========================================
import sys 
import yfinance as yf
import pandas as pd
import numpy as np
from collections import Counter

# ==========================================
# FILE / PROJECT ATTRIBUTION & NOTES
# ==========================================
"""
Project: S&P 500 Stock Outperformance Predictor
Author: Chhayank Kaushik

Note: Since I am using companies that have survived up until today and have not 
gone bankrupt, this introduces survivorship bias, making the model more optimistic.

Data: 100 companies with 9 features from yearly periods (2010–2025).
Train Split: 2010–2019 | Test Split: 2021–2024

References & Credits:
- Decision Tree logic - from Normalized Nerd
- Random Forest ensembling guided by AssemblyAI
"""


tickers = [
    # Mega-cap tech (8)
    "AAPL", "MSFT", "GOOGL", "AMZN", "ORCL", "ADBE", "CRM", "IBM",

    # Semiconductors (6)
    "INTC", "AMD", "TXN", "QCOM", "MU", "AVGO",

    # Communication services (5)
    "NFLX", "DIS", "CMCSA", "T", "VZ",

    # Chronic underperformers / cyclical value, still listed (6)
    "GE", "F", "BA", "WBD", "PFE",

    # Consumer discretionary (10)
    "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "MAR",

    # Consumer staples (8)
    "WMT", "PG", "KO", "PEP", "COST", "CL", "MDLZ", "KMB",

    # Financials (12)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC", "COF",

    # Healthcare (10)
    "JNJ", "UNH", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "CVS", 

    # Industrials (10)
    "CAT", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "UNP", "GD",

    # Energy (7)
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", 

    # Materials (5)
    "LIN", "APD", "ECL", "NEM", "FCX",

    # Utilities (5)
    "NEE", "DUK", "SO", "D", "AEP",

    # REITs (4)
    "SPG", "O", "AMT", "PLD",

    # Meme / high-volatility, long-listed (4)
    "GME", "AMC", "BB", "NOK",

    # Airlines (5)
    "AAL", "UAL", "DAL", "LUV", "JBLU",
]



'''code below gets data for tickers'''

#function to obtain 12-month trailing momentum in a given time-frame
def twelve_month_trailing_momentum(close):    
    momentum_12m = close.shift(21)/close.shift(252) - 1
    # 252 ≈ trading days in 12 months, 21 ≈ trading days in 1 month

    return momentum_12m

#function to obtain annual rolling volatilty of a stock week by week of stock change in past 52 weeks
def annual_rolling_volatility(close):    
    weekly_returns = close.resample('W').last().pct_change()
    annual_volatility = (weekly_returns.rolling(window=52).std()) * np.sqrt(52) #times by sqrt 52 to convert weekly volatilty which has noise into annual volatility

    annual_volatility = annual_volatility.reindex(close.index, method='ffill')#same format as close and moves the volatility to next available time day
    return annual_volatility

#function to see max drawdown, ie largest percentage drop in given timeframe (in days)
def maximum_drawdown(close, window):
    roll_max = close.rolling(window=window).max()
    drawdown = close / roll_max - 1
    max_drawdown = drawdown.rolling(window=window).min()

    return max_drawdown

#func sees how close current price is to 52w high
def high_proximity_52w(close):
    rolling_52w_high = close.rolling(window=252).max()
    proximity_to_high = close / rolling_52w_high

    return proximity_to_high

#func outputs avg trading volume per day over year and also avg trading vol (21d) /avg trading vol (1y)
def trading_vol(vol):
    avg_vol_12m = vol.rolling(window=252).mean()
    vol_trend = vol.rolling(window = 21).mean()/avg_vol_12m

    return avg_vol_12m, vol_trend

#func returns how big company is, it logs to converge outliers so decision tree works better
def log_dollar_vol(close,vol):
    dollar_volume = (close * vol).rolling(window=252).mean()
    log_dollar_volume = np.log(dollar_volume)

    return  log_dollar_volume

def rolling_beta(df,window =252):
    cov = df['stock'].rolling(window).cov(df['sp500'])
    var = df['sp500'].rolling(window).var()

    return cov / var

def future_outperformance(stock_close,sp500_close,window =252):
    stock_return_1y = stock_close.shift(-window) / stock_close  -1 #looking into future hence -ve
    sp500_return_1y = sp500_close.shift(-window) / sp500_close  -1
    sp500_return_1y = sp500_return_1y.reindex(stock_return_1y.index) #ensures they are same length

    label = (stock_return_1y > sp500_return_1y).astype(float)

    label[stock_return_1y.isna() | sp500_return_1y.isna()] = np.nan #ensures Nan > 0.5 is returned at NaN and not false

    return label
    

def build_ticker_table(stock_close,stock_volume,sp500_close,ticker):
    
    stock_returns = stock_close.pct_change()
    SP500_returns = sp500_close.pct_change()

    sp500_stock_returns_df = pd.DataFrame({'stock': stock_returns, 'sp500':SP500_returns}).dropna()

    stock_momentum_12= twelve_month_trailing_momentum(stock_close)
    stock_annual_rolling_volatilty = annual_rolling_volatility(stock_close)
    stock_maximum_drawdown_1y = maximum_drawdown(stock_close,252) #252 is num trading days in year
    stock_maximum_drawdown_3m = maximum_drawdown(stock_close,63) #63 is num trading days in quarter
    stock_high_proximity = high_proximity_52w(stock_close)
    avg_stock_trading_vol, stock_vol_trend = trading_vol(stock_volume)
    stock_log_dollar = log_dollar_vol(stock_close,stock_volume)
    beta_vs_sp500 = rolling_beta(sp500_stock_returns_df)
    stock_outperformance = future_outperformance(stock_close,sp500_close)

    table = pd.DataFrame({
        'momentum_12m': stock_momentum_12,
        'ann_vol': stock_annual_rolling_volatilty,
        'drawdown_1y': stock_maximum_drawdown_1y,
        'drawdown_3m': stock_maximum_drawdown_3m,
        'high_prox_52w': stock_high_proximity,
        'trading_vol': avg_stock_trading_vol,
        'vol_trend': stock_vol_trend,
        'log_dollar_vol': stock_log_dollar,
        'beta': beta_vs_sp500,
        'outperform': stock_outperformance
    })

    table['ticker'] = ticker
    
    
    return table


raw_data= yf.download(tickers + ['SPY'], start="2009-01-01", end="2025-12-31",auto_adjust=True, group_by='ticker', threads=True)
sp500_close = raw_data['SPY']['Close']

all_tables = []

for ticker in tickers:

    stock_close = raw_data[ticker]['Close']
    stock_volume = raw_data[ticker]['Volume']

    table = build_ticker_table(stock_close,stock_volume,sp500_close,ticker)
    table = table.resample('QS').first() #how often we are sampling through data set

    all_tables.append(table)

all_tickers_df = pd.concat(all_tables)
all_tickers_df = all_tickers_df.dropna()
all_tickers_df = all_tickers_df.sort_index()

train_df = all_tickers_df.loc['2010-01-01':'2019-12-31']
test_df = all_tickers_df.loc['2021-01-01':'2024-12-31']

feature_cols = ['momentum_12m', 'ann_vol', 'drawdown_1y', 'drawdown_3m',
                 'high_prox_52w', 'trading_vol', 'vol_trend',
                 'log_dollar_vol', 'beta']

X_train = train_df[feature_cols].values
Y_train = train_df['outperform'].values

X_test = test_df[feature_cols].values
Y_test = test_df['outperform'].values


'''code below makes a decision tree'''


class Node():
    def __init__(self,feature_index = None ,threshold = None, left = None, right = None, info_gain = None, value = None):

        #for the decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain    

        #for the leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2, n_features=None):

        #initialise the root of the tree
        self.root = None

        #stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def build_tree(self, dataset, curr_depth=0):
        #recursive func to build tree

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        #will keep splitting unless stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            #pick random subset of features from main big dataset if n_features is set
            if self.n_features is not None:
                feat_indices = np.random.choice(num_features,self.n_features, replace=False)
            else:
                feat_indices = range(num_features)

            #find best split
            best_split = self.get_best_split(dataset, num_samples, feat_indices)
            #check if info gain is positive to prevent splitting pure nodes
            if best_split.get('info_gain',-1) > 0:
                #recur left
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                #recur right
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                #returns decision node
                return Node(best_split['feature_index'], best_split['threshold'],left_subtree,right_subtree,best_split['info_gain'])

        #compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        #return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, feat_indices, n_thresholds=20):
        #dictionary to store the best split
        best_split = {}
        max_info_gain = -float('inf') #need to use number that is less than any number as starting point

        #loop over all the feautures
        for feature_index in feat_indices:
            feature_values = dataset[:,feature_index]

            #using percentiles instead of every single unique value of feauture, greatly improves speed
            unique_vals = np.unique(feature_values)
            if len(unique_vals) <= n_thresholds:
                possible_thresholds = unique_vals
            else:
                percentiles = np.linspace(0,100, n_thresholds)
                possible_thresholds = np.unique(np.percentile(feature_values,percentiles))
            #loop over all the feature values present in data
            for threshold in possible_thresholds:
                #get current splits
                dataset_left, dataset_right = self.split(dataset,feature_index,threshold)
                #check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:,-1], dataset_left[:,-1],dataset_right[:,-1]
                    #compute gini impurity
                    curr_info_gain = self.information_gain(y, left_y, right_y, 'gini')
                    #update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, dataset, feature_index, threshold):
        #function will split the data

        feature_col = dataset[:,feature_index]
        dataset_left = dataset[feature_col <= threshold]
        dataset_right = dataset[feature_col > threshold]
        return dataset_left,dataset_right

    def information_gain(self, parent, l_child, r_child, mode='entropy'):
        weight_l = len(l_child)/len(parent)
        weight_r = len(r_child)/len(parent)
        if mode == 'gini':
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))

        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        #computes entropy/ note may take lomger as calculating logs takes longer

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y) #fraction how much a certain y value is of total y values 
            entropy += -p_cls *np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        #computes gini_index

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1-gini

    def calculate_leaf_value(self, Y):
        counter = Counter(Y)        
        return counter.most_common(1)[0][0]

    def fit(self, X, Y):
        #function to train the tree
        Y = np.array(Y).reshape(-1,1)
        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        #function to predict new dataset

        predictions = [self.make_prediction(x, self.root) for x in X]  
        return predictions  

    def make_prediction(self, x, tree):
        #function to predict a single data point

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


'''code below will combine multiple decision trees to make random forest'''

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for ii in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                        max_depth=self.max_depth,
                                        n_features=self.n_features
                                        )

            X_sample, y_sample = self.bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)    

    def bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(pred) for pred in tree_predictions])
        return predictions

#evaluate model suite copied and pasted from generative ai as not really point of project but nice to have idea of what alg is doing
def evaluate_model_suite(Y_test, predictions):
    # 1. Confusion Matrix Components
    TP = np.sum((Y_test == 1) & (predictions == 1))
    FP = np.sum((Y_test == 0) & (predictions == 1))
    TN = np.sum((Y_test == 0) & (predictions == 0))
    FN = np.sum((Y_test == 1) & (predictions == 0))

    # 2. Key Ratios
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    

    print(f"--- Classification Diagnostics ---")
    print(f"True Positives (Correct BUYs):   {TP}")
    print(f"False Positives (Failed BUYs):   {FP}")
    print(f"True Negatives (Correct AVOIDs): {TN}")
    print(f"False Negatives (Missed WINs):  {FN}")
    print(f"\n--- Performance Scores ---")
    print(f"Precision (BUY Win Rate): {precision:.2%}")
    print(f"Recall (Winners Caught):  {recall:.2%}")
    print(f"F1-Score:                 {f1:.4f}")
    print(f"Specificity (Avoidance):  {specificity:.2%}")
    

classifier = RandomForestClassifier(
    n_trees=100,
    max_depth=4,
    min_samples_split=20,
    n_features=3
)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

evaluate_model_suite(Y_test, predictions)