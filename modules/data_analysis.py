
import pandas as pd


# ==========================================
# 1. Dataset Preparation
# ==========================================

def prepare_data(df):
    """
    Prepare dataset for business analysis.
    Only Completed orders are used for revenue analysis.
    """

    required_columns = [
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

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
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

    data["Status"] = (
        data["Status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    data["Revenue"] = pd.to_numeric(
        data["Revenue"],
        errors="coerce"
    )

    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    )

    if data["OrderDate"].isna().any():
        raise ValueError("Dataset contains invalid dates.")

    if data[["Revenue", "Quantity"]].isna().any().any():
        raise ValueError(
            "Revenue or Quantity contains invalid values."
        )

    completed = data.loc[
        data["Status"] == "completed"
    ].copy()

    if completed.empty:
        raise ValueError(
            "No completed orders found in the dataset."
        )

    return data, completed


# ==========================================
# 2. Sales Overview
# ==========================================

def sales_overview(df):
    """
    Generate overall business sales KPIs.
    """

    data, completed = prepare_data(df)

    total_orders = int(
        completed["OrderID"].nunique()
    )

    total_customers = int(
        completed["CustomerID"].nunique()
    )

    total_revenue = float(
        completed["Revenue"].sum()
    )

    total_quantity = int(
        completed["Quantity"].sum()
    )

    average_order_value = (
        total_revenue / total_orders
        if total_orders > 0 else 0
    )

    status_counts = (
        data["Status"]
        .value_counts()
        .to_dict()
    )

    status_percentages = (
        data["Status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    return {
        "total_orders": total_orders,
        "total_customers": total_customers,
        "total_revenue": round(total_revenue, 2),
        "total_quantity_sold": total_quantity,
        "average_order_value": round(
            average_order_value, 2
        ),
        "status_counts": status_counts,
        "status_percentages": status_percentages
    }


# ==========================================
# 3. Time Analysis
# ==========================================

def time_analysis(df):
    """
    Analyze daily, monthly and yearly revenue.
    """

    _, completed = prepare_data(df)

    daily = (
        completed.groupby(
            completed["OrderDate"].dt.date
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("OrderID", "nunique"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
        .rename(columns={"OrderDate": "Date"})
    )

    monthly = (
        completed.assign(
            Month=completed["OrderDate"].dt.to_period("M")
        )
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("OrderID", "nunique"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
    )

    monthly["Month"] = (
        monthly["Month"].astype(str)
    )

    yearly = (
        completed.assign(
            Year=completed["OrderDate"].dt.year
        )
        .groupby("Year")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("OrderID", "nunique"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
    )

    if not monthly.empty:

        best_month = monthly.loc[
            monthly["Revenue"].idxmax()
        ].to_dict()

        lowest_month = monthly.loc[
            monthly["Revenue"].idxmin()
        ].to_dict()

    else:
        best_month = None
        lowest_month = None

    return {
        "daily": daily,
        "monthly": monthly,
        "yearly": yearly,
        "best_month": best_month,
        "lowest_month": lowest_month
    }


# ==========================================
# 4. Product Analysis
# ==========================================

def product_analysis(df):
    """
    Analyze product and category performance.
    """

    _, completed = prepare_data(df)

    product_performance = (
        completed.groupby("Product")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity_Sold=("Quantity", "sum"),
            Orders=("OrderID", "nunique")
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_revenue = product_performance[
        "Revenue"
    ].sum()

    if total_revenue > 0:

        product_performance["Revenue_Share"] = (
            product_performance["Revenue"]
            / total_revenue * 100
        ).round(2)

    else:
        product_performance["Revenue_Share"] = 0.0

    category_performance = (
        completed.groupby("Category")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity_Sold=("Quantity", "sum"),
            Orders=("OrderID", "nunique")
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
        .reset_index(drop=True)
    )

    best_product = (
        product_performance.iloc[0].to_dict()
        if not product_performance.empty
        else None
    )

    lowest_product = (
        product_performance.iloc[-1].to_dict()
        if not product_performance.empty
        else None
    )

    return {
        "products": product_performance,
        "categories": category_performance,
        "best_product": best_product,
        "lowest_product": lowest_product,
        "top_5_products": product_performance.head(5)
    }


# ==========================================
# 5. Customer Analysis
# ==========================================

def customer_analysis(df):
    """
    Analyze customer purchasing behavior.
    """

    _, completed = prepare_data(df)

    customer_performance = (
        completed.groupby("CustomerID")
        .agg(
            Total_Orders=("OrderID", "nunique"),
            Total_Revenue=("Revenue", "sum"),
            Total_Quantity=("Quantity", "sum"),
            First_Purchase=("OrderDate", "min"),
            Last_Purchase=("OrderDate", "max")
        )
        .reset_index()
    )

    customer_performance["Average_Order_Value"] = (
        customer_performance["Total_Revenue"]
        / customer_performance["Total_Orders"]
    ).round(2)

    customer_performance["Days_Since_Last_Purchase"] = (
        completed["OrderDate"].max()
        - customer_performance["Last_Purchase"]
    ).dt.days

    customer_performance = (
        customer_performance
        .sort_values(
            "Total_Revenue",
            ascending=False
        )
        .reset_index(drop=True)
    )

    top_customers = (
        customer_performance.head(10)
    )

    return {
        "total_unique_customers": int(
            customer_performance["CustomerID"].nunique()
        ),
        "customer_performance": customer_performance,
        "top_10_customers": top_customers
    }


# ==========================================
# 6. Regional Analysis
# ==========================================

def regional_analysis(df):
    """
    Compare business performance across regions.
    """

    _, completed = prepare_data(df)

    regions = (
        completed.groupby("Region")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("OrderID", "nunique"),
            Quantity_Sold=("Quantity", "sum"),
            Customers=("CustomerID", "nunique")
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
        .reset_index(drop=True)
    )

    best_region = (
        regions.iloc[0].to_dict()
        if not regions.empty
        else None
    )

    lowest_region = (
        regions.iloc[-1].to_dict()
        if not regions.empty
        else None
    )

    return {
        "regions": regions,
        "best_region": best_region,
        "lowest_region": lowest_region
    }


# ==========================================
# 7. Business Summary
# ==========================================

def generate_business_summary(df):
    """
    Generate structured business insights.

    These are calculated by Python and can later
    be passed to the Generative AI module.
    """

    sales = sales_overview(df)
    time = time_analysis(df)
    products = product_analysis(df)
    customers = customer_analysis(df)
    regions = regional_analysis(df)

    summary = {
        "sales_overview": sales,

        "time_analysis": {
            "best_month": time["best_month"],
            "lowest_month": time["lowest_month"]
        },

        "product_analysis": {
            "best_product": products["best_product"],
            "lowest_product": products["lowest_product"]
        },

        "customer_analysis": {
            "total_unique_customers":
                customers["total_unique_customers"],
            "top_customer":
                customers["customer_performance"].iloc[0].to_dict()
                if not customers["customer_performance"].empty
                else None
        },

        "regional_analysis": {
            "best_region": regions["best_region"],
            "lowest_region": regions["lowest_region"]
        }
    }

    return summary


# ==========================================
# 8. Complete Analysis Pipeline
# ==========================================

def analyze_dataset(df):
    """
    Run all business analysis functions.
    """

    return {
        "sales": sales_overview(df),
        "time": time_analysis(df),
        "products": product_analysis(df),
        "customers": customer_analysis(df),
        "regions": regional_analysis(df),
        "business_summary": generate_business_summary(df)
    }