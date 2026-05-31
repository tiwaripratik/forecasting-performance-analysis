# Notebook Explanation: Monthly Price Forecasting

This file explains what happens in `notebook.ipynb`, including the dataset, EDA, train/test split, feature engineering, model parameters, evaluation method, and final forecasting step.

## 1. Notebook Objective

The notebook forecasts monthly average prices from `price_data.csv`.

It compares only these four models:

| Model | Type | Purpose |
|---|---|---|
| ARIMA | Statistical time-series model | Classical baseline for non-stationary price series |
| Random Forest | Machine learning regression | Non-linear lag-based forecasting |
| XGBoost | Machine learning regression | Boosted tree lag-based forecasting |
| LSTM | Deep learning sequence model | Neural network using previous 12 months |

The main methodological rule in this notebook is: during test forecasting, models must not use future actual prices as lag inputs. Random Forest and XGBoost are evaluated recursively, where each future lag is filled using the model's previous prediction.

## 2. Dataset

File used:

```text
price_data.csv
```

Columns:

| Column | Meaning |
|---|---|
| `date` | Monthly date |
| `avg_monthly_price` | Average monthly price |

Dataset summary:

| Property | Value |
|---|---|
| Total rows | 249 |
| Frequency | Monthly start, `MS` |
| Date range | Jan 2005 to Sep 2025 |
| Target column | `avg_monthly_price` |
| Missing values | 0 |
| Minimum price | 3500 |
| Maximum price | 16163 |
| Mean price | 7918.89 |
| Standard deviation | 2804.99 |

The notebook loads the CSV with:

```python
df = pd.read_csv('price_data.csv', parse_dates=['date'])
df = df.set_index('date').asfreq('MS')
prices = df['avg_monthly_price']
```

## 3. Data Cleaning

The cleaning section checks:

| Check | What it verifies |
|---|---|
| Null values | Whether any price/date values are missing |
| Monthly coverage | Whether every expected month exists |
| Duplicate dates | Whether a month appears more than once |
| Duplicate price rows | Repeated price values, not necessarily errors |
| IQR outliers | High/low price values outside the interquartile range |
| Z-score outliers | Extreme points by standard deviation |
| Rolling anomalies | Points far from the 12-month rolling mean |

Important interpretation:

The notebook keeps outliers/anomalies because they look like real market events or regime shifts, not simple data-entry mistakes.

## 4. EDA

The EDA section studies the price series before modeling.

### Trend and Volatility

The notebook creates:

| Feature | Meaning |
|---|---|
| `MA_3` | 3-month moving average |
| `MA_12` | 12-month moving average |
| `MA_24` | 24-month moving average |
| `MoM_%` | Month-over-month percentage change |
| `YoY_%` | Year-over-year percentage change |
| `rolling_vol_12` | 12-month rolling volatility of monthly changes |

These plots show whether the price is trending, whether volatility changes over time, and whether the current regime is different from older periods.

### Regime Analysis

The notebook manually splits history into regimes:

| Regime | Period |
|---|---|
| Stable base | 2005-2008 |
| Boom/decline | 2009-2013 |
| Softening | 2014-2018 |
| Recovery/crash | 2019-2021 |
| Surge | 2022-2025 |

For each regime, it calculates count, mean, standard deviation, min, max, and range.

### Seasonality

The notebook calculates average price by month and monthly standard deviation.

It also computes:

```text
seasonal_index_vs_avg_%
```

This measures how far each month is from the overall average price.

The notebook finds that calendar seasonality is weak compared with structural price shocks.

### STL Decomposition

STL decomposition separates the series into:

| Component | Meaning |
|---|---|
| Trend | Long-term movement |
| Seasonal | Repeating yearly pattern |
| Residual | Unexplained variation/noise |

The notebook calculates trend strength and seasonal strength. The main conclusion is that trend/regime movement matters much more than clean yearly seasonality.

### ACF and PACF

The ACF/PACF plots show autocorrelation.

The series has strong persistence, meaning recent prices are highly informative for near-future prices.

### Additional EDA Added

The newer EDA cells add:

| Analysis | Purpose |
|---|---|
| Distribution percentiles | Understand full price spread and high-price tail |
| Histogram and boxplot | Visualize skew and outliers |
| Last 24/12 months vs full history | Compare recent level with long-term level |
| Year x month heatmap | See regime changes across calendar months |
| Largest MoM and YoY changes | Identify shock months |
| Rolling 12-month mean/std/min/max | Understand recent risk and volatility |
| Distance from 12-month high/low | Show current market position |
| Lag correlations | Identify which historical lags are most predictive |
| Lag-1 scatter plot | Show previous-month persistence |

Main EDA conclusion:

The series is dominated by persistence, long-term regime shifts, and recent high-price behavior. Seasonality exists, but it is weaker than shocks and trend changes.

## 5. Stationarity Tests

The notebook uses:

| Test | Purpose |
|---|---|
| ADF test | Tests whether the series is stationary |
| KPSS test | Tests whether the series is trend-stationary/stationary |

The original series is non-stationary by ADF.

After first differencing, the series becomes stationary. This supports using `d = 1` in ARIMA-style models.

## 6. Train/Test Split

The notebook uses a time-based split.

| Set | Period | Months |
|---|---|---:|
| Training | Jan 2005 to Sep 2024 | 237 |
| Testing | Oct 2024 to Sep 2025 | 12 |

Code:

```python
train = df[:'2024-09-01']
test = df['2024-10-01':'2025-09-01']
y_train = train['avg_monthly_price']
y_test = test['avg_monthly_price']
```

This is correct for time-series forecasting because the model trains only on past data and tests on future data.

## 7. Evaluation Metrics

Each model is evaluated using:

| Metric | Meaning |
|---|---|
| MAE | Average absolute error in price units |
| RMSE | Penalizes larger errors more than MAE |
| MAPE (%) | Average percentage error |
| R2 | How much variance the model explains |

The notebook ranks models mainly by MAPE.

Evaluation function:

```python
evaluate(name, y_true, y_pred, category=None, training_time=None, setup='Recursive')
```

It stores:

| Field | Meaning |
|---|---|
| `Model` | Model name |
| `Category` | Time Series, ML, or Deep Learning |
| `Setup` | Recursive or one-step diagnostic |
| `MAE` | Mean absolute error |
| `RMSE` | Root mean squared error |
| `MAPE (%)` | Percentage error |
| `R2` | R-squared |
| `Training Time (s)` | Runtime where available |

## 8. Feature Engineering for ML Models

Random Forest and XGBoost use engineered features from historical price data.

The feature function is:

```python
create_features(data, lags=[1, 2, 3, 6, 12])
```

It creates 17 features:

| Feature Group | Features |
|---|---|
| Lag prices | `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` |
| Rolling means | `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_12` |
| Rolling volatility | `rolling_std_6`, `rolling_std_12` |
| Calendar | `month`, `quarter`, `year` |
| Cyclical calendar | `month_sin`, `month_cos` |
| Price differences | `price_diff_1`, `price_diff_12` |

Training rows after feature creation:

| Item | Value |
|---|---:|
| Raw training months | 237 |
| Usable ML rows after lag/rolling drops | 225 |
| Feature count | 17 |
| ML feature period | Jan 2006 to Sep 2024 |
| Test months predicted | 12 |

The first 12 months cannot be used for ML training because lag-12 and rolling-12 features need prior data.

## 9. Recursive Forecasting Logic

The notebook uses:

```python
recursive_predict_ml(model, history_df, forecast_dates, feature_cols, scaler=None)
```

For each forecast month:

1. Add the next month to the history.
2. Build lag/rolling/calendar features.
3. Predict the price.
4. Store the prediction.
5. Insert the prediction back into history.
6. Use that predicted value as a lag for later months.

This avoids leakage.

Without recursion, a model might use actual Oct 2024 price to predict Nov 2024, actual Nov 2024 price to predict Dec 2024, and so on. That is useful for one-step diagnostics, but it is too optimistic for a real 12-month forecast.

## 10. Model 1: ARIMA

ARIMA is a statistical time-series model.

Notebook code:

```python
arima_model = auto_arima(
    y_train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    max_p=5,
    max_q=5,
    max_d=2
)
```

Parameters:

| Parameter | Value | Meaning |
|---|---:|---|
| `seasonal` | `False` | Uses non-seasonal ARIMA |
| `stepwise` | `True` | Searches efficiently instead of trying all combinations |
| `suppress_warnings` | `True` | Hides convergence/statistical warnings |
| `max_p` | 5 | Maximum autoregressive order |
| `max_q` | 5 | Maximum moving-average order |
| `max_d` | 2 | Maximum differencing order |

Training data:

| Item | Value |
|---|---:|
| Training months | 237 |
| Training period | Jan 2005 to Sep 2024 |
| Test months | 12 |
| Test period | Oct 2024 to Sep 2025 |

Forecasting:

```python
arima_model.predict(n_periods=len(y_test))
```

ARIMA directly forecasts 12 future months from the fitted time-series model.

## 11. Model 2: Random Forest

Random Forest is a tree-based ensemble regression model.

Notebook code:

```python
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=SEED,
    n_jobs=-1
)
```

Parameters:

| Parameter | Value | Meaning |
|---|---:|---|
| `n_estimators` | 200 | Number of decision trees |
| `max_depth` | 10 | Maximum depth of each tree |
| `random_state` | 42 | Reproducibility seed |
| `n_jobs` | -1 | Use all available CPU cores |

Training data:

| Item | Value |
|---|---:|
| Raw training months | 237 |
| Usable feature rows | 225 |
| Feature count | 17 |
| Training feature period | Jan 2006 to Sep 2024 |
| Test months | 12 |

Prediction setup:

Random Forest is evaluated recursively using `recursive_predict_ml`.

This means the model predicts Oct 2024 first, then uses that predicted Oct 2024 value to build lag features for Nov 2024, and so on through Sep 2025.

Why Random Forest is useful here:

It handles non-linear relationships between lag features, trend, volatility, and current price without requiring the strict stationarity assumptions of ARIMA.

## 12. Model 3: XGBoost

XGBoost is a gradient-boosted tree model.

Notebook code:

```python
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    objective='reg:squarederror'
)
```

Parameters:

| Parameter | Value | Meaning |
|---|---:|---|
| `n_estimators` | 300 | Number of boosting trees |
| `max_depth` | 6 | Maximum depth of each tree |
| `learning_rate` | 0.05 | Step size for each boosting update |
| `subsample` | 0.8 | Uses 80% of rows per tree |
| `colsample_bytree` | 0.8 | Uses 80% of features per tree |
| `random_state` | 42 | Reproducibility seed |
| `objective` | `reg:squarederror` | Squared-error regression objective |

Training data:

| Item | Value |
|---|---:|
| Raw training months | 237 |
| Usable feature rows | 225 |
| Feature count | 17 |
| Training feature period | Jan 2006 to Sep 2024 |
| Test months | 12 |

Prediction setup:

XGBoost is also evaluated recursively with `recursive_predict_ml`.

Why XGBoost is useful here:

It can capture non-linear lag effects and interactions, often more aggressively than Random Forest. It may perform well when recent price momentum and rolling features contain strong predictive signal.

## 13. Model 4: LSTM

LSTM is a recurrent neural network model for sequence data.

Notebook setup:

```python
lookback = 12
scaler_lstm = StandardScaler().fit(train[['avg_monthly_price']].values)
```

Important:

The scaler is fit only on the training data, not the full dataset. This avoids leakage.

Sequence creation:

```python
X_train_lstm, y_train_lstm = create_sequences(train_scaled, lookback)
```

Each LSTM sample uses the previous 12 months to predict the next month.

Training rows:

| Item | Value |
|---|---:|
| Raw training months | 237 |
| Lookback window | 12 months |
| LSTM training sequences | 225 |
| Validation split | 10% of sequences |
| Approx training sequences after split | 203 |
| Approx validation sequences | 22 |
| Test months | 12 |

Architecture:

```python
model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1),
])
```

Layer details:

| Layer | Parameters |
|---|---|
| LSTM 1 | 64 units, returns sequences |
| Dropout 1 | 20% dropout |
| LSTM 2 | 32 units |
| Dropout 2 | 20% dropout |
| Dense hidden | 16 units, ReLU |
| Output | 1 value, predicted price |

Training parameters:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Loss | MSE |
| Epochs | 100 maximum |
| Batch size | 16 |
| Validation split | 0.1 |
| Early stopping patience | 10 |
| Restore best weights | True |

Prediction setup:

The LSTM is evaluated recursively:

1. Start with the last 12 known training prices.
2. Predict the next month.
3. Add that prediction to the sequence.
4. Use the updated sequence to predict the next month.
5. Repeat for all 12 test months.

Why LSTM may struggle:

The dataset has only 249 total monthly points. Deep learning models usually need much more data, so LSTM can overfit or produce unstable forecasts.

## 14. Model Comparison

The notebook creates:

```python
results_df = pd.DataFrame(results).sort_values('MAPE (%)')
```

The model leaderboard includes:

| Model | Category | Setup |
|---|---|---|
| ARIMA | Time Series | Recursive/direct 12-month forecast |
| Random Forest | ML | Recursive |
| XGBoost | ML | Recursive |
| LSTM | Deep Learning | Recursive |

Metrics shown:

| Metric | Lower/Higher Better |
|---|---|
| MAE | Lower |
| RMSE | Lower |
| MAPE | Lower |
| R2 | Higher |

## 15. One-Step Diagnostic Evaluation

The notebook also computes one-step diagnostics for:

| Model |
|---|
| Random Forest |
| XGBoost |

This uses actual earlier test-period values as lag inputs.

Purpose:

It answers: "If we already know the previous month, how well can the model predict the next month?"

But it is not the fair result for a 12-month future forecast, because a real forecast made in Sep 2024 would not know actual Oct 2024, Nov 2024, etc.

The notebook correctly treats this as diagnostic only.

## 16. Final Forecast

After model comparison, the notebook retrains the final Random Forest on all available observed data:

```python
all_features = create_features(df).dropna()
rf_final = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=SEED,
    n_jobs=-1
)
rf_final.fit(all_features[final_feature_cols], all_features['price'])
```

Final training data:

| Item | Value |
|---|---:|
| Full observed months | 249 |
| Full observed period | Jan 2005 to Sep 2025 |
| Usable feature rows after lag/rolling drops | 237 |
| Feature count | 17 |

Forecast horizon:

| Forecast Period | Months |
|---|---:|
| Oct 2025 to Sep 2026 | 12 |

The forecast is generated recursively, again avoiding future leakage.

The notebook then builds:

| Output | Meaning |
|---|---|
| `forecast_df` | Month-wise predicted prices |
| `forecast_summary` | Average, min, max, and trend interpretation |
| Forecast plot | Last 48 months plus next 12 predicted months |

## 17. Business and Deployment Sections

The notebook also includes written sections for:

| Section | Purpose |
|---|---|
| Business actions | How to respond to expected price increases/decreases |
| Measuring action effectiveness | How to track ROI, forecast error, margins, inventory, and procurement savings |
| Django deployment | How to serve/store forecasts in a backend system |
| Django + FastAPI integration | Django for admin/data, FastAPI for low-latency prediction |
| Frontend integration | Next.js / React Native integration ideas |
| Production monitoring | Retraining, drift checks, error tracking, and model governance |

## 18. Important Methodology Notes

1. The notebook uses a time-based split, not random splitting.
2. Random splitting would be wrong for this time-series problem because it would mix future and past records.
3. ML features are lagged/rolling features based on prior information.
4. Recursive evaluation is the correct setup for a true 12-month forecast.
5. One-step diagnostics are useful, but they are intentionally kept separate.
6. The final model is retrained on all observed data before forecasting the next unknown 12 months.

## 19. Summary

The notebook follows this workflow:

1. Load monthly price data.
2. Validate monthly continuity and missing values.
3. Study trend, volatility, regimes, seasonality, shocks, and lag behavior.
4. Test stationarity.
5. Split data into training and test periods.
6. Build lag/rolling/calendar features for ML models.
7. Train ARIMA, Random Forest, XGBoost, and LSTM.
8. Evaluate all models on Oct 2024 to Sep 2025.
9. Compare models using MAE, RMSE, MAPE, and R2.
10. Retrain Random Forest on the full dataset.
11. Forecast Oct 2025 to Sep 2026.
12. Provide business and deployment guidance.

