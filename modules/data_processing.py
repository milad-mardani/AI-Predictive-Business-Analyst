
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 1. Configuration
# ==========================================

REQUIRED_COLUMNS = [
    "OrderID",
    "OrderDate",
    "CustomerID",
    "Product",
    "Category",
    "Quantity",
    "UnitPrice",
    "Revenue",
    "Region",
    "Status"
]

NUMERIC_COLUMNS = [
    "Quantity",
    "UnitPrice",
    "Revenue"
]

IDENTIFIER_COLUMNS = [
    "OrderID",
    "CustomerID",
    "Product",
    "Category",
    "Region",
    "Status"
]


# ==========================================
# 2. Load Dataset
# ==========================================

def load_dataset(source):
    """
    Load CSV or Excel dataset.

    source can be:
    - File path
    - Streamlit UploadedFile
    """

    filename = getattr(source, "name", str(source))
    extension = Path(filename).suffix.lower()

    if hasattr(source, "seek"):
        source.seek(0)

    try:
        if extension == ".csv":

            try:
                df = pd.read_csv(
                    source,
                    encoding="utf-8-sig"
                )

            except UnicodeDecodeError:

                if hasattr(source, "seek"):
                    source.seek(0)

                df = pd.read_csv(
                    source,
                    encoding="cp1252"
                )

        elif extension in [".xlsx", ".xls"]:

            df = pd.read_excel(source)

        else:
            raise ValueError(
                "Unsupported file format. "
                "Please upload CSV or Excel."
            )

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        return df

    except Exception as e:

        raise ValueError(
            f"Could not load dataset: {str(e)}"
        ) from e


# ==========================================
# 3. Normalize Column Names
# ==========================================

def normalize_columns(df):
    """
    Remove unnecessary spaces from column names.
    """

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ==========================================
# 4. Validate Required Columns
# ==========================================

def validate_columns(df):
    """
    Check whether required business columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


# ==========================================
# 5. Dataset Preview
# ==========================================

def preview_data(df, rows=10):
    """
    Return first rows of dataset.
    """

    return df.head(rows)


# ==========================================
# 6. Data Quality Inspection
# ==========================================

def inspect_data(df):
    """
    Generate basic data quality information.
    """

    missing_values = df.isnull().sum()

    missing_percentage = (
        missing_values / len(df) * 100
    ).round(2)

    missing_report = pd.DataFrame({

        "Missing Values": missing_values,

        "Missing Percentage": missing_percentage

    })

    data_types = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],

        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]

    })

    report = {

        "total_rows": len(df),

        "total_columns": len(df.columns),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "missing_values": missing_report,

        "data_types": data_types,

        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum()
            / (1024 ** 2),
            2
        )

    }

    return report


# ==========================================
# 7. Data Cleaning
# ==========================================

def clean_data(df):
    """
    Clean a copy of the original dataset.

    Original dataframe remains unchanged.
    """

    df = normalize_columns(df)

    validate_columns(df)

    original_rows = len(df)

    report = {}

    # --------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    report["duplicates_removed"] = duplicate_count

    # --------------------------------------
    # Clean text columns
    # --------------------------------------

    for column in IDENTIFIER_COLUMNS:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # Empty strings become missing values
    df = df.replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )

    # --------------------------------------
    # Convert date column
    # --------------------------------------

    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        errors="coerce"
    )

    invalid_dates = int(
        df["OrderDate"].isna().sum()
    )

    report["invalid_dates"] = invalid_dates

    # --------------------------------------
    # Convert numeric columns
    # --------------------------------------

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------
    # Remove rows with missing identifiers
    # --------------------------------------

    critical_columns = [
        "OrderID",
        "OrderDate",
        "CustomerID",
        "Product"
    ]

    missing_critical = int(
        df[critical_columns].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=critical_columns
    ).copy()

    report["missing_critical_rows_removed"] = (
        missing_critical
    )

    # --------------------------------------
    # Remove invalid quantities and prices
    # --------------------------------------

    invalid_numeric_mask = (

        df["Quantity"].isna()

        | df["UnitPrice"].isna()

        | (df["Quantity"] <= 0)

        | (df["UnitPrice"] < 0)

    )

    invalid_numeric_count = int(
        invalid_numeric_mask.sum()
    )

    df = df.loc[
        ~invalid_numeric_mask
    ].copy()

    report["invalid_numeric_rows_removed"] = (
        invalid_numeric_count
    )

    # --------------------------------------
    # Fill missing Revenue
    # --------------------------------------

    missing_revenue = df["Revenue"].isna()

    missing_revenue_count = int(
        missing_revenue.sum()
    )

    df.loc[
        missing_revenue,
        "Revenue"
    ] = (

        df.loc[missing_revenue, "Quantity"]

        * df.loc[missing_revenue, "UnitPrice"]

    )

    report["missing_revenue_filled"] = (
        missing_revenue_count
    )

    # --------------------------------------
    # Fill missing categorical values
    # --------------------------------------

    categorical_columns = [
        "Category",
        "Region",
        "Status"
    ]

    for column in categorical_columns:

        df[column] = df[column].fillna(
            "Unknown"
        )

    # --------------------------------------
    # Final formatting
    # --------------------------------------

    df["Quantity"] = df["Quantity"].astype(int)

    df["Revenue"] = df["Revenue"].round(2)

    df["UnitPrice"] = df["UnitPrice"].round(2)

    df = df.reset_index(drop=True)

    report["original_rows"] = original_rows

    report["cleaned_rows"] = len(df)

    report["rows_removed"] = (
        original_rows - len(df)
    )

    report["remaining_missing_values"] = int(
        df.isna().sum().sum()
    )

    return df, report


# ==========================================
# 8. Data Validation
# ==========================================

def validate_data(df):
    """
    Validate business data consistency.
    """

    validate_columns(df)

    revenue_difference = (
        df["Revenue"]
        - df["Quantity"] * df["UnitPrice"]
    ).abs()

    report = {

        "total_rows": len(df),

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "invalid_dates": int(
            df["OrderDate"].isna().sum()
        ),

        "invalid_quantities": int(
            (df["Quantity"] <= 0).sum()
        ),

        "negative_prices": int(
            (df["UnitPrice"] < 0).sum()
        ),

        "revenue_mismatches": int(
            (revenue_difference > 0.01).sum()
        ),

        "revenue_validation": bool(
            (revenue_difference <= 0.01).all()
        )

    }

    report["is_valid"] = all([

        report["missing_values"] == 0,

        report["duplicate_rows"] == 0,

        report["invalid_dates"] == 0,

        report["invalid_quantities"] == 0,

        report["negative_prices"] == 0,

        report["revenue_validation"]

    ])

    return report


# ==========================================
# 9. Date Feature Engineering
# ==========================================

def add_date_features(df):
    """
    Extract useful time-based features.
    """

    df = df.copy()

    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        errors="coerce"
    )

    df["OrderYear"] = df["OrderDate"].dt.year

    df["OrderMonth"] = df["OrderDate"].dt.month

    df["OrderDay"] = df["OrderDate"].dt.day

    df["DayOfWeek"] = df["OrderDate"].dt.dayofweek

    df["DayName"] = df["OrderDate"].dt.day_name()

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    return df


# ==========================================
# 10. Customer Feature Engineering
# ==========================================

def create_customer_features(
    df,
    reference_date=None
):
    """
    Generate customer-level purchasing features.

    Only Completed orders are considered.

    reference_date:
        The date used to calculate recency.
    """

    completed = df.loc[
        df["Status"].str.lower() == "completed"
    ].copy()

    if completed.empty:

        raise ValueError(
            "No completed orders available "
            "for customer feature engineering."
        )

    completed["OrderDate"] = pd.to_datetime(
        completed["OrderDate"],
        errors="coerce"
    )

    if reference_date is None:

        reference_date = (
            completed["OrderDate"].max()
            + pd.Timedelta(days=1)
        )

    else:

        reference_date = pd.Timestamp(
            reference_date
        )

    # --------------------------------------
    # Customer aggregation
    # --------------------------------------

    customers = completed.groupby(
        "CustomerID"
    ).agg(

        TotalOrders=(
            "OrderID",
            "nunique"
        ),

        TotalRevenue=(
            "Revenue",
            "sum"
        ),

        FirstPurchase=(
            "OrderDate",
            "min"
        ),

        LastPurchase=(
            "OrderDate",
            "max"
        )

    ).reset_index()

    # --------------------------------------
    # Additional features
    # --------------------------------------

    customers["AverageOrderValue"] = (

        customers["TotalRevenue"]

        / customers["TotalOrders"]

    ).round(2)

    customers["DaysSinceLastPurchase"] = (

        reference_date

        - customers["LastPurchase"]

    ).dt.days

    active_days = (

        customers["LastPurchase"]

        - customers["FirstPurchase"]

    ).dt.days.clip(lower=1)

    customers["PurchaseFrequency"] = (

        (customers["TotalOrders"] - 1)

        / active_days

        * 30

    ).round(4)

    # RFM-compatible columns

    customers["Recency"] = (
        customers["DaysSinceLastPurchase"]
    )

    customers["Frequency"] = (
        customers["TotalOrders"]
    )

    customers["Monetary"] = (
        customers["TotalRevenue"]
    )

    customers["TotalRevenue"] = (
        customers["TotalRevenue"].round(2)
    )

    return customers


# ==========================================
# 11. Complete Processing Pipeline
# ==========================================

def process_dataset(source):
    """
    Complete basic data processing pipeline.

    Returns:
        cleaned dataframe
        inspection report
        cleaning report
        validation report
    """

    original_df = load_dataset(source)

    original_df = normalize_columns(
        original_df
    )

    validate_columns(original_df)

    inspection_report = inspect_data(
        original_df
    )

    cleaned_df, cleaning_report = clean_data(
        original_df
    )

    validation_report = validate_data(
        cleaned_df
    )

    return {

        "data": cleaned_df,

        "inspection": inspection_report,

        "cleaning": cleaning_report,

        "validation": validation_report

    }