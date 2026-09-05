# S&P 500 Outperformance Predictor

A custom **Random Forest Classifier built from scratch** in Python to predict whether individual stocks will outperform the S&P 500 over a 12-month forward window. 

Rather than relying on `scikit-learn` for core modeling, this project implements low-level algorithm architecture—including Gini Impurity calculations, recursive node splitting, and bootstrap aggregation.

---

## Technical Highlights & Architecture

* **From-Scratch Algorithm Design:** Custom `Node`, `DecisionTreeClassifier`, and `RandomForestClassifier` classes featuring Gini impurity split selection, recursive tree generation, and bootstrap ensembling with majority voting.
* **Leakage-Free Time-Series Split:** Strictly enforced chronological separation using a `2010–2019` training window and a `2021–2024` out-of-sample testing window and one year gap between test and training periods to eliminate lookahead bias.
* **Multi-Factor Feature Pipeline:** Feature set combining momentum, volatility, downside risk, and liquidity metrics extracted from raw market data:
  * **Momentum:** 12-Month Trailing Momentum, Proximity to 52-Week High.
  * **Volatility & Risk:** Annual Rolling Volatility, Yearly/Quarterly Max Drawdown, Rolling Market Beta vs. `SPY`.
  * **Liquidity & Volume Trends:** Average Daily Volume (1-Year), Relative Volume (RVOL), Log Dollar Volume.

---

## Model Evaluation & Diagnostics

To assess real-world financial utility, model outputs were evaluated using precision, recall, specificity, and F1-score on the unseen `2021–2024` test set:

```text
--- Classification Diagnostics ---
True Positives (Correct BUYs):   168
False Positives (Failed BUYs):   195
True Negatives (Correct AVOIDs): 718
False Negatives (Missed WINs):   519

--- Performance Scores ---
Precision (BUY Win Rate): 46.28%
Recall (Winners Caught):  24.45%
F1-Score:                 0.3200
Specificity (Avoidance):  78.64%
```

## Backtest Analysis & Diagnostic Takeaways

### 1. Risk-Averse Filtering Strength
The model achieved a **78.64% Specificity**, correctly identifying and avoiding 718 out of 913 underperforming assets. The Random Forest acts primarily as a strong risk-aversion filter, it excels at screening out loser stock, it struggles to isolate high conviction winners.

### 2. Conservative Hyperparameters (`min_samples_split = 20`)
Setting `min_samples_split = 20` effectively prevented the trees from overfitting to noise in individual stock charts. However, this high threshold also made the ensemble overly conservative so subtle, low-frequency buy signals were ignored, contributing to the low **Recall** and **Precision**.

---

## Quantitative Limitations & Real-World Risks

* **Macroeconomic Regime Shift:** The model was trained on 2010–2019 data, a period defined by near-zero interest rates, low inflation, and sustained mega-cap tech outperformance. The 2021–2024 test period broke from this pattern in two distinct phases: a sharp Fed rate hiking cycle and rotation into energy/value stocks in 2022, followed by a reversal back toward mega-cap tech/growth in 2023–2024 driven by the AI rally. Because the model's decision thresholds were calibrated on the stable, low-rate conditions of the 2010s, they do not generalize well to this more volatile, regime-shifting period
* **Survivorship Bias:** The stock universe used is built from tickers that are actively trading today. Companies that went bankrupt, were acquired, or were delisted between 2010–2024 are not included, which inflates historical performance and risk metrics relative to what would have been experienced in real time. This is a limitation of how the universe is constructed.

---

## Acknowledgements

* **Normalized Nerd** — Reference for low-level `DecisionTreeClassifier` recursive node construction and Gini impurity splitting logic.
* **AssemblyAI** — Architecture guidelines for bootstrapping datasets and assembling the `RandomForestClassifier`
