
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. Configuration
# ==========================================

LAG_DAYS = [1, 7, 14, 28]

ROLLING_WINDOWS = [7, 14, 28]

RANDOM_STATE = 42


# ==========================================
# 2. Prepare Daily Sales Data
# ==========================================

def prepare_daily_sales(df):
    """
    Create daily revenue time series.

    Only Completed orders are considered.
    Days without sales are filled with zero.
    """

    required_columns = [
        "OrderDate",
        "Revenue",
        "Status"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df.copy()

    data["OrderDate"] = pd.to_datetime(
        data["OrderDate"],
        errors="coerce"
    )

    data["Revenue"] = pd.to_numeric(
        data["Revenue"],
        errors="coerce"
    )

    data["Status"] = (
        data["Status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if data["OrderDate"].isna().any():
        raise ValueError(
            "Invalid dates found in dataset."
        )

    if data["Revenue"].isna().any():
        raise ValueError(
            "Invalid revenue values found in dataset."
        )

    completed = data.loc[
        data["Status"] == "completed"
    ].copy()

    if completed.empty:
        raise ValueError(
            "No completed orders found."
        )

    daily_sales = (
        completed.groupby(
            completed["OrderDate"].dt.normalize()
        )["Revenue"]
        .sum()
        .sort_index()
    )

    full_date_range = pd.date_range(
        start=daily_sales.index.min(),
        end=daily_sales.index.max(),
        freq="D"
    )

    daily_sales = (
        daily_sales
        .reindex(full_date_range, fill_value=0)
        .astype(float)
    )

    daily_sales.index.name = "Date"

    return daily_sales


# ==========================================
# 3. Create Time-Based Features
# ==========================================

def create_features(history, target_date):
    """
    Generate features using only historical values.
    """

    history = np.asarray(
        history,
        dtype=float
    )

    if len(history) < max(LAG_DAYS):
        raise ValueError(
            "At least 28 days of history are required."
        )

    target_date = pd.Timestamp(target_date)

    features = {}

    # Lag features
    for lag in LAG_DAYS:

        features[f"Lag_{lag}"] = float(
            history[-lag]
        )

    # Rolling statistics
    for window in ROLLING_WINDOWS:

        recent_values = history[-window:]

        features[f"Rolling_Mean_{window}"] = float(
            np.mean(recent_values)
        )

        features[f"Rolling_Std_{window}"] = float(
            np.std(recent_values)
        )

    # Calendar features
    features["DayOfWeek"] = target_date.dayofweek

    features["DayOfMonth"] = target_date.day

    features["Month"] = target_date.month

    features["DayOfYear"] = target_date.dayofyear

    features["IsWeekend"] = int(
        target_date.dayofweek >= 5
    )

    return features


# ==========================================
# 4. Build Training Dataset
# ==========================================

def build_training_data(daily_sales):
    """
    Convert time series into supervised ML data.
    """

    values = daily_sales.values

    dates = daily_sales.index

    features_list = []

    targets = []

    for i in range(max(LAG_DAYS), len(values)):

        features = create_features(
            values[:i],
            dates[i]
        )

        features_list.append(features)

        targets.append(values[i])

    X = pd.DataFrame(features_list)

    y = np.asarray(
        targets,
        dtype=float
    )

    return X, y


# ==========================================
# 5. Create ML Model
# ==========================================

def create_model():
    """
    Create Random Forest regression model.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return model


# ==========================================
# 6. Forecast Future Values
# ==========================================

def predict_future(
    model,
    historical_values,
    start_date,
    forecast_days=30
):
    """
    Generate recursive daily forecasts.

    Each prediction is added to history
    and used to predict the following day.
    """

    history = list(
        np.asarray(
            historical_values,
            dtype=float
        )
    )

    start_date = pd.Timestamp(start_date)

    predictions = []

    for day in range(forecast_days):

        target_date = (
            start_date
            + pd.Timedelta(days=day)
        )

        features = create_features(
            history,
            target_date
        )

        X_future = pd.DataFrame(
            [features]
        )

        prediction = float(
            model.predict(X_future)[0]
        )

        prediction = max(0.0, prediction)

        predictions.append({
            "Date": target_date,
            "Predicted_Revenue": round(
                prediction, 2
            )
        })

        history.append(prediction)

    return pd.DataFrame(predictions)


# ==========================================
# 7. Calculate Evaluation Metrics
# ==========================================

def calculate_metrics(actual, predicted):
    """
    Evaluate forecasting performance.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    total_actual = np.sum(
        np.abs(actual)
    )

    wape = (
        np.sum(np.abs(actual - predicted))
        / total_actual * 100
        if total_actual > 0
        else None
    )

    return {
        "MAE": round(float(mae), 2),

        "RMSE": round(float(rmse), 2),

        "WAPE_Percent": (
            round(float(wape), 2)
            if wape is not None
            else None
        )
    }


# ==========================================
# 8. Train and Evaluate Model
# ==========================================

def evaluate_forecasting_model(
    daily_sales,
    test_days=30
):
    """
    Chronological holdout evaluation.

    The model predicts the final test_days
    recursively without using actual test values.
    """

    if test_days < 1:
        raise ValueError(
            "test_days must be at least 1."
        )

    split_index = len(daily_sales) - test_days

    if split_index <= max(LAG_DAYS):
        raise ValueError(
            "Not enough historical data for evaluation."
        )

    values = daily_sales.values

    dates = daily_sales.index

    X_train = []

    y_train = []

    for i in range(max(LAG_DAYS), split_index):

        X_train.append(
            create_features(
                values[:i],
                dates[i]
            )
        )

        y_train.append(
            values[i]
        )

    X_train = pd.DataFrame(X_train)

    y_train = np.asarray(
        y_train,
        dtype=float
    )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    # Recursive predictions for the test period
    test_predictions = predict_future(
        model=model,
        historical_values=values[:split_index],
        start_date=dates[split_index],
        forecast_days=test_days
    )

    actual_values = values[split_index:]

    predicted_values = (
        test_predictions["Predicted_Revenue"].values
    )

    metrics = calculate_metrics(
        actual_values,
        predicted_values
    )

    evaluation_results = pd.DataFrame({

        "Date": dates[split_index:],

        "Actual_Revenue": actual_values,

        "Predicted_Revenue": predicted_values

    })

    return {
        "model": model,
        "metrics": metrics,
        "predictions": evaluation_results
    }


# ==========================================
# 9. Train Final Model
# ==========================================

def train_final_model(daily_sales):
    """
    Train model using all available historical data.
    """

    X, y = build_training_data(
        daily_sales
    )

    model = create_model()

    model.fit(X, y)

    return model


# ==========================================
# 10. Complete Sales Forecasting Pipeline
# ==========================================

def run_sales_forecasting(
    df,
    forecast_days=30,
    test_days=30
):
    """
    Complete sales forecasting pipeline.

    Returns:
        daily historical sales
        model evaluation metrics
        test predictions
        future forecast
        trained model
    """

    if forecast_days < 1:
        raise ValueError(
            "forecast_days must be at least 1."
        )

    daily_sales = prepare_daily_sales(df)

    if len(daily_sales) <= max(LAG_DAYS) + test_days:
        raise ValueError(
            "Not enough historical data for forecasting."
        )

    # Evaluate forecasting performance
    evaluation = evaluate_forecasting_model(
        daily_sales,
        test_days=test_days
    )

    # Train final model on all historical data
    final_model = train_final_model(
        daily_sales
    )

    # Forecast future revenue
    forecast_start = (
        daily_sales.index.max()
        + pd.Timedelta(days=1)
    )

    future_forecast = predict_future(
        model=final_model,
        historical_values=daily_sales.values,
        start_date=forecast_start,
        forecast_days=forecast_days
    )

    return {
        "historical_sales": daily_sales,

        "evaluation_metrics": evaluation["metrics"],

        "evaluation_predictions": evaluation["predictions"],

        "forecast": future_forecast,

        "model": final_model,

        "forecast_days": forecast_days,

        "last_historical_date": str(
            daily_sales.index.max().date()
        )
    }
