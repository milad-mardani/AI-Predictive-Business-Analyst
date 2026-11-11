import pandas as pd
import numpy as np

from modules.sales_forecasting import run_sales_forecasting
from modules.demand_forecasting import run_demand_forecasting
from modules.churn_prediction import run_churn_prediction
from modules.customer_segmentation import run_customer_segmentation


# =========================================================
# 1. Configuration
# =========================================================

FORECAST_DAYS = 30
TEST_DAYS = 30
CHURN_DAYS = 90
N_CLUSTERS = 4
TOP_CHURN_CUSTOMERS = 5


# =========================================================
# 2. Convert DataFrame to JSON-Friendly Records
# =========================================================

def dataframe_to_records(df, limit=None):

    if df is None or df.empty:
        return []

    data = df.copy()

    if limit is not None:
        data = data.head(limit)

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.astype(object).where(
        pd.notna(data),
        None
    )

    return data.to_dict(orient="records")


# =========================================================
# 3. Prepare Export DataFrame
# =========================================================

def prepare_export_dataframe(results, key, analysis_name):

    if key not in results:
        raise KeyError(
            f"{analysis_name}: '{key}' was not found in model results."
        )

    data = results[key]

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{analysis_name}: '{key}' must be a pandas DataFrame."
        )

    return data.copy()


# =========================================================
# 4. Dataset Overview
# =========================================================

def build_dataset_overview(df):

    required_columns = [
        "OrderDate",
        "CustomerID",
        "Product",
        "Revenue",
        "Quantity",
        "Status"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    data = df.copy()

    data["OrderDate"] = pd.to_datetime(
        data["OrderDate"],
        errors="coerce"
    )

    completed = data[
        data["Status"].astype(str).str.strip().str.lower()
        == "completed"
    ]

    return {

        "dataset_type": "Synthetic E-commerce Dataset",

        "total_transactions": int(len(data)),

        "unique_customers": int(
            data["CustomerID"].nunique()
        ),

        "unique_products": int(
            data["Product"].nunique()
        ),

        "completed_transactions": int(
            len(completed)
        ),

        "date_range": {

            "start": str(
                data["OrderDate"].min().date()
            ),

            "end": str(
                data["OrderDate"].max().date()
            )
        }
    }


# =========================================================
# 5. Sales Forecasting Context
# =========================================================

def build_sales_context(df):

    results = run_sales_forecasting(
        df,
        forecast_days=FORECAST_DAYS,
        test_days=TEST_DAYS
    )

    forecast = results["forecast"]

    metrics = results["evaluation_metrics"]

    required_forecast_columns = [
        "Date",
        "Predicted_Revenue"
    ]

    missing = [
        col for col in required_forecast_columns
        if col not in forecast.columns
    ]

    if missing:
        raise ValueError(
            f"Sales forecast is missing columns: {missing}"
        )

    if forecast.empty:
        raise ValueError(
            "Sales forecasting returned an empty forecast."
        )

    forecast = forecast.copy()

    forecast["Date"] = pd.to_datetime(
        forecast["Date"],
        errors="coerce"
    )

    daily_forecast = []

    for _, row in forecast.iterrows():

        daily_forecast.append({

            "date": str(row["Date"].date()),

            "predicted_revenue": round(
                float(row["Predicted_Revenue"]),
                2
            )
        })

    context = {

        "forecast_period_days": results["forecast_days"],

        "last_historical_date": (
            str(results["last_historical_date"])
        ),

        "model": "Random Forest Regressor",

        "forecast_summary": {

            "predicted_total_revenue": round(
                float(
                    forecast["Predicted_Revenue"].sum()
                ),
                2
            ),

            "average_daily_revenue": round(
                float(
                    forecast["Predicted_Revenue"].mean()
                ),
                2
            ),

            "minimum_daily_revenue": round(
                float(
                    forecast["Predicted_Revenue"].min()
                ),
                2
            ),

            "maximum_daily_revenue": round(
                float(
                    forecast["Predicted_Revenue"].max()
                ),
                2
            ),

            "forecast_start_date": str(
                forecast["Date"].min().date()
            ),

            "forecast_end_date": str(
                forecast["Date"].max().date()
            ),

            "daily_forecast": daily_forecast
        },

        "evaluation_metrics": metrics
    }

    export_data = prepare_export_dataframe(
        results,
        "forecast",
        "Sales Forecasting"
    )

    return context, export_data


# =========================================================
# 6. Product Demand Forecasting Context
# =========================================================

def build_demand_context(df):

    results = run_demand_forecasting(
        df,
        forecast_days=FORECAST_DAYS,
        test_days=TEST_DAYS
    )

    summary = results["forecast_summary"]

    evaluation = results["evaluation_metrics"]

    context = {

        "forecast_period_days": results["forecast_days"],

        "model": "Random Forest Regressor",

        "products_evaluated": int(
            evaluation["Products_Evaluated"]
        ),

        "products_skipped": int(
            evaluation["Products_Skipped"]
        ),

        "overall_evaluation_metrics": evaluation,

        "top_products_by_predicted_demand": (
            dataframe_to_records(
                summary,
                limit=10
            )
        ),

        "skipped_products": dataframe_to_records(
            results["skipped_products"],
            limit=10
        )
    }

    export_data = prepare_export_dataframe(
        results,
        "product_forecasts",
        "Product Demand Forecasting"
    )

    return context, export_data


# =========================================================
# 7. Customer Churn Context
# =========================================================

def build_churn_context(df):

    results = run_churn_prediction(
        df,
        churn_days=CHURN_DAYS
    )

    customer_predictions = prepare_export_dataframe(
        results,
        "customer_predictions",
        "Customer Churn Prediction"
    )

    required_columns = [
        "CustomerID",
        "Churn_Probability",
        "Risk_Level"
    ]

    missing = [
        col for col in required_columns
        if col not in customer_predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Customer predictions are missing columns: {missing}"
        )

    if customer_predictions.empty:
        raise ValueError(
            "Customer churn prediction returned no customer records."
        )

    customer_predictions = customer_predictions.copy()

    customer_predictions["Churn_Probability"] = pd.to_numeric(
        customer_predictions["Churn_Probability"],
        errors="coerce"
    )

    if customer_predictions["Churn_Probability"].isna().any():
        raise ValueError(
            "Customer churn probabilities contain invalid values."
        )

    total_customers = len(customer_predictions)

    # -----------------------------------------------------
    # Risk Distribution and Percentages
    # -----------------------------------------------------

    risk_summary = prepare_export_dataframe(
        results,
        "risk_summary",
        "Customer Churn Prediction"
    )

    if not {"Risk_Level", "Customer_Count"}.issubset(
        risk_summary.columns
    ):
        raise ValueError(
            "Risk summary must contain Risk_Level and Customer_Count."
        )

    risk_summary = risk_summary.copy()

    risk_summary["Percentage"] = (
        risk_summary["Customer_Count"] / total_customers * 100
    ).round(2)

    risk_distribution = dataframe_to_records(
        risk_summary
    )

    # -----------------------------------------------------
    # High Risk Customer Statistics
    # -----------------------------------------------------

    high_risk_customers = customer_predictions[
        customer_predictions["Risk_Level"].astype(str).str.strip()
        == "High Risk"
    ]

    high_risk_count = len(high_risk_customers)

    high_risk_percentage = round(
        (high_risk_count / total_customers) * 100,
        2
    )

    # -----------------------------------------------------
    # Top 5 Customers by Churn Probability
    # -----------------------------------------------------

    top_customers = customer_predictions.sort_values(
        by="Churn_Probability",
        ascending=False
    ).head(TOP_CHURN_CUSTOMERS)

    top_customers = top_customers.copy()

    top_customers["Churn_Probability"] = (
        top_customers["Churn_Probability"].round(6)
    )

    top_customers_records = dataframe_to_records(
        top_customers
    )

    # -----------------------------------------------------
    # AI Context
    # -----------------------------------------------------

    context = {

        "churn_definition": (
            "A customer is labeled as churned if "
            "they make no completed purchase during "
            "the following 90-day observation window."
        ),

        "churn_days": results["churn_days"],

        "reference_date": str(
            pd.Timestamp(
                results["reference_date"]
            ).date()
        ),

        "total_customers_evaluated": int(
            total_customers
        ),

        "high_risk_customer_count": int(
            high_risk_count
        ),

        "high_risk_customer_percentage": (
            high_risk_percentage
        ),

        "risk_distribution": risk_distribution,

        "top_5_customers_by_churn_probability": (
            top_customers_records
        ),

        "evaluation_metrics": (
            results["evaluation_metrics"]
        ),

        "confusion_matrix": (
            results["confusion_matrix"]
        ),

        "training_rows": results["training_rows"],

        "train_rows": results["train_rows"],

        "test_rows": results["test_rows"],

        "top_features": dataframe_to_records(
            results["feature_importance"],
            limit=5
        )
    }

    export_data = customer_predictions

    return context, export_data


# =========================================================
# 8. Customer Segmentation Context
# =========================================================

def build_segmentation_context(df):

    results = run_customer_segmentation(
        df,
        n_clusters=N_CLUSTERS
    )

    context = {

        "method": "RFM + K-Means",

        "n_clusters": results["n_clusters"],

        "total_customers": results["total_customers"],

        "reference_date": str(
            pd.Timestamp(
                results["reference_date"]
            ).date()
        ),

        "segment_summary": dataframe_to_records(
            results["segment_summary"]
        ),

        "business_insights": dataframe_to_records(
            results["business_insights"]
        )
    }

    export_data = prepare_export_dataframe(
        results,
        "rfm_data",
        "Customer Segmentation"
    )

    return context, export_data


# =========================================================
# 9. Build Complete Business Context
# =========================================================

def build_business_context(df):

    if df is None or df.empty:
        raise ValueError(
            "Dataset is empty. Cannot build business context."
        )

    print("\nBuilding Business Context...")

    print("\n[1/5] Dataset Overview...")

    dataset_context = build_dataset_overview(df)

    print("[2/5] Running Sales Forecasting...")

    sales_context, sales_export = build_sales_context(df)

    print("[3/5] Running Product Demand Forecasting...")

    demand_context, demand_export = build_demand_context(df)

    print("[4/5] Running Customer Churn Prediction...")

    churn_context, churn_export = build_churn_context(df)

    print("[5/5] Running Customer Segmentation...")

    segmentation_context, segmentation_export = (
        build_segmentation_context(df)
    )

    business_context = {

        "dataset_overview": dataset_context,

        "sales_forecasting": sales_context,

        "product_demand_forecasting": demand_context,

        "customer_churn_prediction": churn_context,

        "customer_segmentation": segmentation_context,

        "data_limitations": [

            "The dataset contains synthetic e-commerce data.",

            "Forecasts are model predictions, not guaranteed outcomes.",

            "Churn risk represents model-estimated risk, not confirmed churn.",

            "Customer segments are interpretations of K-Means clusters.",

            "Model evaluation metrics are based on historical holdout data."
        ],

        # Full model outputs for CSV exports.
        # Do not send this section to the Generative AI model.

        "export_data": {

            "sales_forecast": sales_export,

            "product_demand": demand_export,

            "customer_churn": churn_export,

            "customer_segments": segmentation_export
        }
    }

    print("\nBusiness Context Successfully Created.")

    print("\nExport Data Summary:")

    for name, export_df in business_context["export_data"].items():

        print(
            f"{name}: {len(export_df):,} rows, "
            f"{len(export_df.columns)} columns"
        )

    return business_context


# =========================================================
# 10. Prepare Context for Generative AI
# =========================================================

def get_ai_business_context(business_context):

    """
    Remove full export DataFrames before sending context
    to the Generative AI model.

    Compact sales forecast and customer churn information
    are included in their respective contexts.
    """

    return {
        key: value
        for key, value in business_context.items()
        if key != "export_data"
    }