
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# CONFIGURATION
# ==================================================

CHURN_DAYS = 90
RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 2

RISK_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40
}


# ==================================================
# 1. DATA PREPARATION
# ==================================================

def prepare_churn_data(df):

    required_columns = [
        "OrderDate",
        "CustomerID",
        "OrderID",
        "Quantity",
        "Revenue",
        "Status"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
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

    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "OrderDate",
            "CustomerID",
            "OrderID",
            "Revenue",
            "Quantity"
        ]
    )

    data = data[
        (data["Quantity"] > 0)
        & (data["Revenue"] >= 0)
    ].copy()

    data = data[
        data["Status"].astype(str).str.strip().str.lower()
        == "completed"
    ].copy()

    if data.empty:
        raise ValueError(
            "No valid Completed orders found."
        )

    data["OrderDate"] = data["OrderDate"].dt.normalize()

    return data


# ==================================================
# 2. CUSTOMER FEATURES
# ==================================================

def create_customer_features(
    data,
    reference_date
):

    reference_date = pd.Timestamp(
        reference_date
    ).normalize()

    history = data[
        data["OrderDate"] < reference_date
    ].copy()

    if history.empty:
        return pd.DataFrame()

    features = history.groupby(
        "CustomerID"
    ).agg(
        Total_Orders=("OrderID", "nunique"),
        Total_Revenue=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        First_Order_Date=("OrderDate", "min"),
        Last_Order_Date=("OrderDate", "max"),
        Unique_Order_Days=("OrderDate", "nunique")
    )

    features["Days_Since_Last_Order"] = (
        reference_date - features["Last_Order_Date"]
    ).dt.days

    features["Customer_Tenure_Days"] = (
        reference_date - features["First_Order_Date"]
    ).dt.days

    features["Average_Order_Value"] = (
        features["Total_Revenue"]
        / features["Total_Orders"].clip(lower=1)
    )

    features["Average_Quantity_Per_Order"] = (
        features["Total_Quantity"]
        / features["Total_Orders"].clip(lower=1)
    )

    # Average time between unique order dates
    features["Average_Days_Between_Orders"] = (
        features["Customer_Tenure_Days"]
        / (features["Unique_Order_Days"] - 1).clip(lower=1)
    )

    # Purchase frequency
    features["Orders_Per_30_Days"] = (
        features["Total_Orders"]
        / (
            features["Customer_Tenure_Days"].clip(lower=1)
            / 30
        )
    )

    return features.reset_index()


# ==================================================
# 3. CREATE HISTORICAL CHURN LABELS
# ==================================================

def create_churn_training_data(
    data,
    churn_days=CHURN_DAYS,
    min_history_days=90
):

    first_date = data["OrderDate"].min()
    last_date = data["OrderDate"].max()

    # Each snapshot needs enough history and
    # a complete future churn observation window.
    first_snapshot = (
        first_date + pd.Timedelta(days=min_history_days)
    )

    last_snapshot = (
        last_date - pd.Timedelta(days=churn_days)
    )

    if first_snapshot > last_snapshot:
        raise ValueError(
            "Insufficient historical data for churn labeling."
        )

    snapshot_dates = pd.date_range(
        first_snapshot,
        last_snapshot,
        freq="30D"
    )

    training_rows = []

    for snapshot_date in snapshot_dates:

        customer_features = create_customer_features(
            data,
            snapshot_date
        )

        if customer_features.empty:
            continue

        future_end = (
            snapshot_date
            + pd.Timedelta(days=churn_days)
        )

        future_orders = data[
            (data["OrderDate"] >= snapshot_date)
            & (data["OrderDate"] < future_end)
        ]

        future_active_customers = set(
            future_orders["CustomerID"].unique()
        )

        customer_features["Churn"] = (
            ~customer_features["CustomerID"].isin(
                future_active_customers
            )
        ).astype(int)

        customer_features["Snapshot_Date"] = (
            snapshot_date
        )

        training_rows.append(
            customer_features
        )

    if not training_rows:
        raise ValueError(
            "Could not create churn training samples."
        )

    training_data = pd.concat(
        training_rows,
        ignore_index=True
    )

    return training_data


# ==================================================
# 4. MODEL FEATURES
# ==================================================

FEATURE_COLUMNS = [
    "Total_Orders",
    "Total_Revenue",
    "Total_Quantity",
    "Days_Since_Last_Order",
    "Customer_Tenure_Days",
    "Average_Order_Value",
    "Average_Quantity_Per_Order",
    "Average_Days_Between_Orders",
    "Orders_Per_30_Days"
]


def prepare_model_features(data):

    X = data[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    return X


# ==================================================
# 5. CREATE MODEL
# ==================================================

def create_churn_model():

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ==================================================
# 6. TRAIN / TEST SPLIT
# ==================================================

def chronological_split(
    training_data,
    test_size=0.20
):

    data = training_data.sort_values(
        "Snapshot_Date"
    ).copy()

    unique_dates = sorted(
        data["Snapshot_Date"].unique()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "Not enough snapshot dates for chronological evaluation."
        )

    split_position = max(
        1,
        int(len(unique_dates) * (1 - test_size))
    )

    split_position = min(
        split_position,
        len(unique_dates) - 1
    )

    split_date = unique_dates[split_position]

    train = data[
        data["Snapshot_Date"] < split_date
    ].copy()

    test = data[
        data["Snapshot_Date"] >= split_date
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Chronological train/test split produced an empty set."
        )

    if train["Churn"].nunique() < 2:
        raise ValueError(
            "Training data contains only one churn class."
        )

    return train, test


# ==================================================
# 7. MODEL EVALUATION
# ==================================================

def evaluate_churn_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "Precision": round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ), 4
        ),
        "Recall": round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ), 4
        ),
        "F1_Score": round(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ), 4
        )
    }

    if y_test.nunique() == 2:
        metrics["ROC_AUC"] = round(
            roc_auc_score(
                y_test,
                probabilities
            ), 4
        )
    else:
        metrics["ROC_AUC"] = None

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    confusion = {
        "True_Negatives": int(matrix[0, 0]),
        "False_Positives": int(matrix[0, 1]),
        "False_Negatives": int(matrix[1, 0]),
        "True_Positives": int(matrix[1, 1])
    }

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "predictions": predictions,
        "probabilities": probabilities
    }


# ==================================================
# 8. CUSTOMER RISK CLASSIFICATION
# ==================================================

def assign_risk_level(probability):

    if probability >= RISK_THRESHOLDS["high"]:
        return "High Risk"

    if probability >= RISK_THRESHOLDS["medium"]:
        return "Medium Risk"

    return "Low Risk"


# ==================================================
# 9. FEATURE IMPORTANCE
# ==================================================

def get_feature_importance(model):

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    return importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)


# ==================================================
# 10. RUN CHURN PREDICTION
# ==================================================

def run_churn_prediction(
    df,
    churn_days=CHURN_DAYS
):

    # Prepare completed orders
    data = prepare_churn_data(df)

    # Create historical training samples
    training_data = create_churn_training_data(
        data,
        churn_days=churn_days
    )

    if training_data["Churn"].nunique() < 2:
        raise ValueError(
            "Training data must contain both churn classes."
        )

    # Chronological evaluation
    train_data, test_data = chronological_split(
        training_data
    )

    X_train = prepare_model_features(
        train_data
    )

    y_train = train_data["Churn"]

    X_test = prepare_model_features(
        test_data
    )

    y_test = test_data["Churn"]

    if y_test.nunique() < 2:
        raise ValueError(
            "Test data must contain both churn classes. "
            "More historical data or a different evaluation split is needed."
        )

    evaluation_model = create_churn_model()

    evaluation_model.fit(
        X_train,
        y_train
    )

    evaluation = evaluate_churn_model(
        evaluation_model,
        X_test,
        y_test
    )

    # Train final model using all historical samples
    final_model = create_churn_model()

    final_model.fit(
        prepare_model_features(training_data),
        training_data["Churn"]
    )

    # Current customer features
    reference_date = (
        data["OrderDate"].max()
        + pd.Timedelta(days=1)
    )

    current_features = create_customer_features(
        data,
        reference_date
    )

    if current_features.empty:
        raise ValueError(
            "Could not generate current customer features."
        )

    X_current = prepare_model_features(
        current_features
    )

    current_probabilities = final_model.predict_proba(
        X_current
    )[:, 1]

    customer_predictions = current_features[
        [
            "CustomerID",
            "Total_Orders",
            "Total_Revenue",
            "Average_Order_Value",
            "Days_Since_Last_Order",
            "Customer_Tenure_Days"
        ]
    ].copy()

    customer_predictions["Churn_Probability"] = (
        current_probabilities
    )

    customer_predictions["Risk_Level"] = (
        customer_predictions["Churn_Probability"]
        .apply(assign_risk_level)
    )

    customer_predictions = (
        customer_predictions.sort_values(
            "Churn_Probability",
            ascending=False
        ).reset_index(drop=True)
    )

    # Summary
    risk_summary = (
        customer_predictions["Risk_Level"]
        .value_counts()
        .reindex(
            [
                "High Risk",
                "Medium Risk",
                "Low Risk"
            ],
            fill_value=0
        )
        .rename_axis("Risk_Level")
        .reset_index(name="Customer_Count")
    )

    return {
        "training_data": training_data,
        "evaluation_metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "evaluation_predictions": evaluation["predictions"],
        "evaluation_probabilities": evaluation["probabilities"],
        "test_labels": y_test.reset_index(drop=True),
        "final_model": final_model,
        "customer_predictions": customer_predictions,
        "risk_summary": risk_summary,
        "feature_importance": get_feature_importance(
            final_model
        ),
        "reference_date": reference_date,
        "churn_days": churn_days,
        "training_rows": len(training_data),
        "train_rows": len(train_data),
        "test_rows": len(test_data)
    }