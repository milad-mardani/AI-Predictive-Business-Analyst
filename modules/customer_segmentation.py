
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==================================================
# CONFIGURATION
# ==================================================

N_CLUSTERS = 4
RANDOM_STATE = 42


# ==================================================
# 1. PREPARE DATA
# ==================================================

def prepare_segmentation_data(df):

    required_columns = [
        "OrderDate",
        "CustomerID",
        "OrderID",
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

    data = data.dropna(
        subset=[
            "OrderDate",
            "CustomerID",
            "OrderID",
            "Revenue"
        ]
    )

    data = data[
        (data["Revenue"] >= 0)
        & (data["CustomerID"].astype(str).str.strip() != "")
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
# 2. CALCULATE RFM FEATURES
# ==================================================

def calculate_rfm(data):

    reference_date = (
        data["OrderDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = data.groupby(
        "CustomerID"
    ).agg(
        Last_Order_Date=("OrderDate", "max"),
        Frequency=("OrderID", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    rfm["Recency"] = (
        reference_date - rfm["Last_Order_Date"]
    ).dt.days

    rfm["Average_Order_Value"] = (
        rfm["Monetary"]
        / rfm["Frequency"].clip(lower=1)
    )

    rfm = rfm[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "Average_Order_Value"
        ]
    ].copy()

    rfm = rfm.replace(
        [np.inf, -np.inf],
        np.nan
    )

    rfm = rfm.dropna(
        subset=[
            "Recency",
            "Frequency",
            "Monetary"
        ]
    )

    rfm = rfm[
        (rfm["Recency"] >= 0)
        & (rfm["Frequency"] > 0)
        & (rfm["Monetary"] >= 0)
    ].copy()

    if rfm.empty:
        raise ValueError(
            "No valid RFM customer records found."
        )

    return rfm, reference_date


# ==================================================
# 3. PREPARE FEATURES FOR K-MEANS
# ==================================================

RFM_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary"
]


def prepare_clustering_features(rfm):

    features = rfm[RFM_COLUMNS].copy()

    # Log transformation reduces skewness
    features["Recency"] = np.log1p(
        features["Recency"]
    )

    features["Frequency"] = np.log1p(
        features["Frequency"]
    )

    features["Monetary"] = np.log1p(
        features["Monetary"]
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.fillna(0)

    return features


# ==================================================
# 4. ASSIGN BUSINESS-FRIENDLY SEGMENT NAMES
# ==================================================

def assign_segment_names(rfm):

    # Higher scores are better.
    # Recency is reversed because lower recency is better.
    rfm = rfm.copy()

    rfm["R_Score"] = (
        1 - rfm["Recency"].rank(pct=True)
    )

    rfm["F_Score"] = (
        rfm["Frequency"].rank(pct=True)
    )

    rfm["M_Score"] = (
        rfm["Monetary"].rank(pct=True)
    )

    rfm["RFM_Score"] = (
        0.40 * rfm["R_Score"]
        + 0.30 * rfm["F_Score"]
        + 0.30 * rfm["M_Score"]
    )

    cluster_scores = (
        rfm.groupby("Cluster")["RFM_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    segment_names = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At Risk"
    ]

    # The number of names matches the actual clusters.
    name_mapping = {
        cluster: segment_names[index]
        for index, cluster in enumerate(
            cluster_scores.index
        )
    }

    rfm["Segment"] = rfm["Cluster"].map(
        name_mapping
    )

    return rfm


# ==================================================
# 5. GENERATE SEGMENT SUMMARY
# ==================================================

def generate_segment_summary(rfm):

    summary = rfm.groupby(
        ["Cluster", "Segment"]
    ).agg(
        Customer_Count=("CustomerID", "nunique"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean"),
        Average_Order_Value=("Average_Order_Value", "mean"),
        Average_RFM_Score=("RFM_Score", "mean")
    ).reset_index()

    summary["Customer_Percent"] = (
        summary["Customer_Count"]
        / summary["Customer_Count"].sum()
        * 100
    )

    summary = summary.sort_values(
        "Average_RFM_Score",
        ascending=False
    )

    return summary.round(2).reset_index(drop=True)


# ==================================================
# 6. GENERATE BUSINESS INSIGHTS
# ==================================================

def generate_segmentation_insights(
    summary
):

    insights = []

    for _, row in summary.iterrows():

        segment = row["Segment"]
        count = int(row["Customer_Count"])

        if segment == "Champions":
            description = (
                "Customers with strong recent purchase "
                "activity and high purchase value."
            )

        elif segment == "Loyal Customers":
            description = (
                "Customers with relatively strong "
                "purchase frequency and value."
            )

        elif segment == "Potential Loyalists":
            description = (
                "Customers who may have potential "
                "for stronger engagement."
            )

        else:
            description = (
                "Customers with weaker RFM profiles "
                "who may need attention."
            )

        insights.append({
            "Segment": segment,
            "Customer_Count": count,
            "Description": description
        })

    return pd.DataFrame(insights)


# ==================================================
# 7. RUN CUSTOMER SEGMENTATION
# ==================================================

def run_customer_segmentation(
    df,
    n_clusters=N_CLUSTERS
):

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    # Prepare transactions
    data = prepare_segmentation_data(df)

    # Calculate RFM
    rfm, reference_date = calculate_rfm(data)

    if len(rfm) < n_clusters:
        raise ValueError(
            "Number of customers is smaller "
            "than the requested number of clusters."
        )

    # Prepare clustering features
    X = prepare_clustering_features(rfm)

    # Scale features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Train K-Means
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    rfm["Cluster"] = model.fit_predict(
        X_scaled
    )

    # Assign business-friendly names
    rfm = assign_segment_names(rfm)

    # Generate summaries
    segment_summary = generate_segment_summary(
        rfm
    )

    business_insights = generate_segmentation_insights(
        segment_summary
    )

    return {
        "rfm_data": rfm,
        "segment_summary": segment_summary,
        "business_insights": business_insights,
        "model": model,
        "scaler": scaler,
        "reference_date": reference_date,
        "n_clusters": n_clusters,
        "total_customers": len(rfm)
    }