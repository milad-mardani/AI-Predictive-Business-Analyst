
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

MIN_HISTORY_DAYS = 90

MIN_NONZERO_DAYS = 10


# ==========================================
# 2. Prepare Daily Product Demand
# ==========================================

def prepare_daily_demand(df):
    """
    Create a daily demand time series for each product.

    Only Completed orders are considered.
    Days without sales are filled with zero.
    """

    required_columns = [
        "OrderDate",
        "Product",
        "Quantity",
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

    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    )

    data["Product"] = (
        data["Product"]
        .astype("string")
        .str.strip()
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

    if data["Quantity"].isna().any():
        raise ValueError(
            "Invalid quantity values found in dataset."
        )

    if (data["Quantity"] <= 0).any():
        raise ValueError(
            "Quantity must be greater than zero."
        )

    if data["Product"].isna().any() or (
        data["Product"] == ""
    ).any():
        raise ValueError(
            "Missing or empty product names found."
        )

    completed = data.loc[
        data["Status"] == "completed"
    ].copy()

    if completed.empty:
        raise ValueError(
            "No completed orders found."
        )

    completed["OrderDate"] = (
        completed["OrderDate"].dt.normalize()
    )

    # Use one common date range for all products.
    start_date = data["OrderDate"].min().normalize()

    end_date = data["OrderDate"].max().normalize()

    full_date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    products = sorted(
        data["Product"].unique().tolist()
    )

    daily_demand = {}

    for product in products:

        product_orders = completed.loc[
            completed["Product"] == product
        ]

        product_daily = (
            product_orders
            .groupby("OrderDate")["Quantity"]
            .sum()
            .sort_index()
        )

        product_daily = (
            product_daily
            .reindex(
                full_date_range,
                fill_value=0
            )
            .astype(float)
        )

        product_daily.index.name = "Date"

        daily_demand[product] = product_daily

    return daily_demand


# ==========================================
# 3. Create Time-Based Features
# ==========================================

def create_features(history, target_date):
    """
    Generate features using only historical demand values.
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

def build_training_data(daily_demand):
    """
    Convert one product's time series into supervised ML data.
    """

    values = daily_demand.values

    dates = daily_demand.index

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
# 6. Forecast Future Demand
# ==========================================

def predict_future(
    model,
    historical_values,
    start_date,
    forecast_days=30
):
    """
    Generate recursive daily demand forecasts.

    Each prediction is added to history
    and used to predict the following day.
    """

    if forecast_days < 1:
        raise ValueError(
            "forecast_days must be at least 1."
        )

    history = list(
        np.asarray(
            historical_values,
            dtype=float
        )
    )

    if len(history) < max(LAG_DAYS):
        raise ValueError(
            "Not enough historical data for forecasting."
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

        # Demand cannot be negative.
        # Convert predicted units to whole numbers.
        prediction = max(
            0,
            int(round(prediction))
        )

        predictions.append({
            "Date": target_date,
            "Predicted_Demand": prediction
        })

        history.append(float(prediction))

    return pd.DataFrame(predictions)


# ==========================================
# 7. Seasonal Naive Baseline
# ==========================================

def predict_seasonal_naive(
    historical_values,
    forecast_days=30
):
    """
    Predict future demand using the same weekday
    from the previous week.
    """

    history = list(
        np.asarray(
            historical_values,
            dtype=float
        )
    )

    if len(history) < 7:
        raise ValueError(
            "At least 7 days of history are required."
        )

    predictions = []

    for _ in range(forecast_days):

        prediction = max(
            0,
            int(round(history[-7]))
        )

        predictions.append(prediction)

        history.append(float(prediction))

    return predictions


# ==========================================
# 8. Calculate Evaluation Metrics
# ==========================================

def calculate_metrics(actual, predicted):
    """
    Evaluate demand forecasting performance.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    if len(actual) == 0:
        raise ValueError(
            "Actual values cannot be empty."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted arrays must have equal lengths."
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
# 9. Evaluate One Product
# ==========================================

def evaluate_product(
    daily_demand,
    test_days=30
):
    """
    Chronological holdout evaluation for one product.

    The test period is predicted recursively without
    using actual test values.
    """

    if test_days < 1:
        raise ValueError(
            "test_days must be at least 1."
        )

    split_index = (
        len(daily_demand) - test_days
    )

    if split_index <= max(LAG_DAYS):
        raise ValueError(
            "Not enough historical data for evaluation."
        )

    values = daily_demand.values

    dates = daily_demand.index

    train_values = values[:split_index]

    # Build training data using only the training period.
    training_series = pd.Series(
        train_values,
        index=dates[:split_index]
    )

    X_train, y_train = build_training_data(
        training_series
    )

    if len(X_train) < 30:
        raise ValueError(
            "Not enough training samples."
        )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    # Random Forest predictions
    test_predictions = predict_future(
        model=model,
        historical_values=train_values,
        start_date=dates[split_index],
        forecast_days=test_days
    )

    actual_values = values[split_index:]

    predicted_values = (
        test_predictions["Predicted_Demand"].values
    )

    # Seasonal Naive baseline
    baseline_predictions = predict_seasonal_naive(
        historical_values=train_values,
        forecast_days=test_days
    )

    # Evaluation metrics
    rf_metrics = calculate_metrics(
        actual_values,
        predicted_values
    )

    baseline_metrics = calculate_metrics(
        actual_values,
        baseline_predictions
    )

    evaluation_results = pd.DataFrame({

        "Date": dates[split_index:],

        "Actual_Demand": actual_values,

        "Predicted_Demand": predicted_values,

        "Seasonal_Naive": baseline_predictions

    })

    return {
        "model": model,

        "rf_metrics": rf_metrics,

        "baseline_metrics": baseline_metrics,

        "predictions": evaluation_results
    }


# ==========================================
# 10. Train Final Model
# ==========================================

def train_final_model(daily_demand):
    """
    Train a product model using all available history.
    """

    X, y = build_training_data(
        daily_demand
    )

    model = create_model()

    model.fit(
        X,
        y
    )

    return model


# ==========================================
# 11. Complete Demand Forecasting Pipeline
# ==========================================

def run_demand_forecasting(
    df,
    forecast_days=30,
    test_days=30
):
    """
    Complete product demand forecasting pipeline.

    Each product receives an independent Random Forest model.

    Returns:
        historical_demand
        evaluation_metrics
        product_evaluation_metrics
        evaluation_predictions
        product_forecasts
        forecast_summary
        trained_models
        skipped_products
    """

    if forecast_days < 1:
        raise ValueError(
            "forecast_days must be at least 1."
        )

    if test_days < 1:
        raise ValueError(
            "test_days must be at least 1."
        )

    # Prepare daily demand for all products
    all_demand = prepare_daily_demand(df)

    trained_models = {}

    product_evaluations = []

    evaluation_predictions = []

    product_forecasts = []

    skipped_products = []

    rf_actual_all = []

    rf_predicted_all = []

    baseline_predicted_all = []

    # Process each product independently
    for product, daily_demand in all_demand.items():

        nonzero_days = int(
            (daily_demand > 0).sum()
        )

        if len(daily_demand) < MIN_HISTORY_DAYS:

            skipped_products.append({
                "Product": product,
                "Reason": "Insufficient historical days"
            })

            continue

        if nonzero_days < MIN_NONZERO_DAYS:

            skipped_products.append({
                "Product": product,
                "Reason": "Insufficient non-zero demand days"
            })

            continue

        if len(daily_demand) <= max(LAG_DAYS) + test_days:

            skipped_products.append({
                "Product": product,
                "Reason": "Insufficient data for evaluation"
            })

            continue

        try:

            # Evaluate model
            evaluation = evaluate_product(
                daily_demand,
                test_days=test_days
            )

            product_evaluations.append({

                "Product": product,

                "Random_Forest_MAE": (
                    evaluation["rf_metrics"]["MAE"]
                ),

                "Random_Forest_RMSE": (
                    evaluation["rf_metrics"]["RMSE"]
                ),

                "Random_Forest_WAPE_Percent": (
                    evaluation["rf_metrics"]["WAPE_Percent"]
                ),

                "Seasonal_Naive_MAE": (
                    evaluation["baseline_metrics"]["MAE"]
                ),

                "Seasonal_Naive_RMSE": (
                    evaluation["baseline_metrics"]["RMSE"]
                ),

                "Seasonal_Naive_WAPE_Percent": (
                    evaluation["baseline_metrics"]["WAPE_Percent"]
                )

            })

            product_test_results = (
                evaluation["predictions"].copy()
            )

            product_test_results["Product"] = product

            evaluation_predictions.append(
                product_test_results
            )

            rf_actual_all.extend(
                product_test_results["Actual_Demand"].tolist()
            )

            rf_predicted_all.extend(
                product_test_results["Predicted_Demand"].tolist()
            )

            baseline_predicted_all.extend(
                product_test_results["Seasonal_Naive"].tolist()
            )

            # Train final model on all available history
            final_model = train_final_model(
                daily_demand
            )

            trained_models[product] = final_model

            # Forecast future demand
            forecast_start = (
                daily_demand.index.max()
                + pd.Timedelta(days=1)
            )

            future_forecast = predict_future(
                model=final_model,

                historical_values=daily_demand.values,

                start_date=forecast_start,

                forecast_days=forecast_days
            )

            future_forecast["Product"] = product

            product_forecasts.append(
                future_forecast
            )

        except ValueError as error:

            skipped_products.append({
                "Product": product,
                "Reason": str(error)
            })

    if not product_forecasts:

        raise ValueError(
            "No products had sufficient data for forecasting."
        )

    # Combine all product forecasts
    forecast_df = pd.concat(
        product_forecasts,
        ignore_index=True
    )

    forecast_df = forecast_df[
        [
            "Date",
            "Product",
            "Predicted_Demand"
        ]
    ]

    # Product-level 30-day summary
    forecast_summary = (
        forecast_df
        .groupby("Product")
        .agg(
            Total_Predicted_Demand=(
                "Predicted_Demand",
                "sum"
            ),

            Average_Daily_Demand=(
                "Predicted_Demand",
                "mean"
            ),

            Maximum_Daily_Demand=(
                "Predicted_Demand",
                "max"
            )
        )
        .reset_index()
        .sort_values(
            "Total_Predicted_Demand",
            ascending=False
        )
        .reset_index(drop=True)
    )

    forecast_summary[
        "Average_Daily_Demand"
    ] = forecast_summary[
        "Average_Daily_Demand"
    ].round(2)

    # Combine evaluation results
    evaluation_df = pd.concat(
        evaluation_predictions,
        ignore_index=True
    )

    product_metrics_df = pd.DataFrame(
        product_evaluations
    )

    # Aggregate metrics across all evaluated products
    overall_rf_metrics = calculate_metrics(
        rf_actual_all,
        rf_predicted_all
    )

    overall_baseline_metrics = calculate_metrics(
        rf_actual_all,
        baseline_predicted_all
    )

    evaluation_metrics = {
        "Random_Forest": overall_rf_metrics,

        "Seasonal_Naive": overall_baseline_metrics,

        "Products_Evaluated": len(trained_models),

        "Products_Skipped": len(skipped_products)
    }

    return {
        "historical_demand": all_demand,

        "evaluation_metrics": evaluation_metrics,

        "product_evaluation_metrics": product_metrics_df,

        "evaluation_predictions": evaluation_df,

        "product_forecasts": forecast_df,

        "forecast_summary": forecast_summary,

        "models": trained_models,

        "skipped_products": pd.DataFrame(
            skipped_products,
            columns=["Product", "Reason"]
        ),

        "forecast_days": forecast_days,

        "test_days": test_days
    }
