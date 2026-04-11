# Volatility Forecasting & Targeting with HAR + XGBoost
This project develops a volatility forecasting and portfolio allocation framework using machine learning and econometric models. Our goal is to determine whether improved volatility forecasts can result in better risk-adjusted investment performance.

## Overview
In this project, we will be comparing three different volatility forecasting approaches:

- **Naive (persistence)**
- **HAR model** (multi-horizon volatility dynamics)
- **HAR + XGBoost** (machine learning model on HAR residuals)

While the HAR-model captures volatility dynamics well, XGBoost is well-able to obtain signal from nonlinear data easily. We use these facts to hypothesize that HAR + XGB will perform strongly compared to our other approaches.
Our forecasts will be used in a **volatility-targeting strategy**, where portfolio exposure is dynamically scaled based on predicted risk.

## Methodology
### Volatility Forecasting
The HAR+XGB model combines a linear HAR baseline with gradient boosted machine learning:

$$\hat{y}_t^{HAR+XGB} = \hat{y}_t^{HAR} + \hat{\varepsilon}_t^{XGB},$$

where $\hat{y}_t^{HAR}$ is the HAR prediction and $\hat{\varepsilon}_t^{XGB}$ denotes the residual estimated by XGBoost.

To achieve optimal performance, our model hyperparameters are tuned using Bayesian optimization (Optuna).
Models are evaluated using an **expanding-window walk-forward scheme**, in order to ensure no look-ahead bias occurs in our time-series data. The performances are evaluated using a forecasting (signal) stability algorithm across different periods and an out-of-sample analysis.

### Volatility Targeting
To implement in our volatility targeting algorithm, we define the portfolio leverage $L_t$:

$$L_t = \frac{\sigma^*}{\hat{\sigma}_{t,\text{ann}}}$$

- $\sigma^*$: target annualized volatility  
- $\hat{\sigma}_{t,\text{ann}}$: predicted volatility  

Returns are computed using lagged leverage, and **transaction costs** are included to simulate a real-world implementation of our models.

## Results
- HAR+XGB improves forecast accuracy over HAR:
  - RMSE -7.3%, MAE -4.4%, QLIKE -6.0%

- In the volatility-targeting strategy:
  - **HAR+XGB achieves the highest Sharpe ($0.46$)**
  - Outperforms HAR ($0.42$) and naive ($0.37$)
  - Slightly exceeds Buy & Hold ($0.43$) after costs

- Without transaction costs:
  - Sharpe increases to **$0.54$**
  - All strategies outperform Buy & Hold

**Key takeaway:** ML forecasting improvements are economically profitable, but trading costs can reduce this profit marginally.

## Conclusion
- Our volatility targeting algorithm **reduces risk** (lower volatility and drawdowns)
- The HAR model provides a **strong baseline**
- Machine learning combined with HAR yields **incremental improvements**
- Economic gain is **sensitive to transaction costs**

## Project Structure
src/
- data.py # data loading
- features.py # feature engineering
- models.py # HAR & XGBoost models
- backtest.py # backtesting algorithm
- plots.py # visualization

main_pipeline.ipynb # main pipeline notebook
