
import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.business_context import (
    build_business_context,
    get_ai_business_context
)

from modules.ai_business_analyst import ask_business_question


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Predictive Business Analyst",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "sample_data.csv"

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


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def prepare_dataset(df):

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        errors="coerce"
    )

    if df["OrderDate"].isna().any():
        raise ValueError(
            "Some OrderDate values are invalid."
        )

    numeric_columns = [
        "Quantity",
        "UnitPrice",
        "Revenue"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Invalid or missing numeric values detected."
        )

    return df


def create_source_key(df, source_name):

    dataset_hash = pd.util.hash_pandas_object(
        df,
        index=True
    ).values.tobytes()

    return hashlib.md5(
        source_name.encode() + dataset_hash
    ).hexdigest()


def dataframe_to_csv_bytes(df):

    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


def build_business_report(context):

    overview = context["dataset_overview"]

    sales = context["sales_forecasting"]

    demand = context["product_demand_forecasting"]

    churn = context["customer_churn_prediction"]

    segmentation = context["customer_segmentation"]

    lines = []

    lines.append("=" * 65)
    lines.append("AI PREDICTIVE BUSINESS ANALYST")
    lines.append("BUSINESS ANALYSIS REPORT")
    lines.append("=" * 65)

    lines.append("")
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 40)

    lines.append(
        f"Dataset Type: {overview['dataset_type']}"
    )

    lines.append(
        f"Total Transactions: {overview['total_transactions']:,}"
    )

    lines.append(
        f"Unique Customers: {overview['unique_customers']:,}"
    )

    lines.append(
        f"Unique Products: {overview['unique_products']:,}"
    )

    lines.append(
        f"Completed Transactions: {overview['completed_transactions']:,}"
    )

    lines.append(
        f"Date Range: {overview['date_range']['start']} "
        f"to {overview['date_range']['end']}"
    )

    # -----------------------------------------------------
    # SALES FORECAST
    # -----------------------------------------------------

    lines.append("")
    lines.append("2. SALES FORECASTING")
    lines.append("-" * 40)

    forecast = sales["forecast_summary"]

    metrics = sales["evaluation_metrics"]

    lines.append(
        f"Forecast Period: {sales['forecast_period_days']} days"
    )

    lines.append(
        f"Model: {sales['model']}"
    )

    lines.append(
        f"Predicted Total Revenue: "
        f"{forecast['predicted_total_revenue']:,.2f}"
    )

    lines.append(
        f"Average Daily Revenue: "
        f"{forecast['average_daily_revenue']:,.2f}"
    )

    lines.append(
        f"Minimum Daily Revenue: "
        f"{forecast['minimum_daily_revenue']:,.2f}"
    )

    lines.append(
        f"Maximum Daily Revenue: "
        f"{forecast['maximum_daily_revenue']:,.2f}"
    )

    lines.append(
        f"MAE: {metrics.get('MAE', 'N/A')}"
    )

    lines.append(
        f"RMSE: {metrics.get('RMSE', 'N/A')}"
    )

    lines.append(
        f"WAPE: {metrics.get('WAPE_Percent', 'N/A')}%"
    )

    # -----------------------------------------------------
    # PRODUCT DEMAND
    # -----------------------------------------------------

    lines.append("")
    lines.append("3. PRODUCT DEMAND FORECASTING")
    lines.append("-" * 40)

    lines.append(
        f"Forecast Period: {demand['forecast_period_days']} days"
    )

    lines.append(
        f"Products Evaluated: {demand['products_evaluated']}"
    )

    lines.append(
        f"Products Skipped: {demand['products_skipped']}"
    )

    lines.append("")
    lines.append("Top Products by Predicted Demand:")

    for product in demand["top_products_by_predicted_demand"]:

        lines.append(
            f"- {product['Product']}: "
            f"{product['Total_Predicted_Demand']} units"
        )

    # -----------------------------------------------------
    # CUSTOMER CHURN
    # -----------------------------------------------------

    lines.append("")
    lines.append("4. CUSTOMER CHURN PREDICTION")
    lines.append("-" * 40)

    lines.append(
        f"Churn Definition: {churn['churn_definition']}"
    )

    lines.append(
        f"Observation Window: {churn['churn_days']} days"
    )

    for metric, value in churn["evaluation_metrics"].items():

        lines.append(
            f"{metric}: {value}"
        )

    lines.append("")
    lines.append("Customer Risk Distribution:")

    for risk in churn["risk_distribution"]:

        lines.append(
            f"- {risk['Risk_Level']}: "
            f"{risk['Customer_Count']} customers"
        )

    # -----------------------------------------------------
    # CUSTOMER SEGMENTATION
    # -----------------------------------------------------

    lines.append("")
    lines.append("5. CUSTOMER SEGMENTATION")
    lines.append("-" * 40)

    lines.append(
        f"Method: {segmentation['method']}"
    )

    lines.append(
        f"Number of Segments: {segmentation['n_clusters']}"
    )

    lines.append(
        f"Total Customers: {segmentation['total_customers']}"
    )

    for segment in segmentation["segment_summary"]:

        lines.append(
            f"- {segment['Segment']}: "
            f"{segment['Customer_Count']} customers "
            f"({segment['Customer_Percent']}%)"
        )

    # -----------------------------------------------------
    # LIMITATIONS
    # -----------------------------------------------------

    lines.append("")
    lines.append("6. DATA AND MODEL LIMITATIONS")
    lines.append("-" * 40)

    for limitation in context["data_limitations"]:

        lines.append(f"- {limitation}")

    lines.append("")
    lines.append("=" * 65)
    lines.append("END OF REPORT")
    lines.append("=" * 65)

    return "\n".join(lines)


def display_evaluation_metrics(metrics):

    if not metrics:
        st.info("Evaluation metrics are not available.")
        return

    metric_names = [
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("WAPE_Percent", "WAPE (%)")
    ]

    columns = st.columns(3)

    for i, (key, label) in enumerate(metric_names):

        if key in metrics:

            columns[i].metric(
                label,
                f"{metrics[key]:,.2f}"
            )


# =========================================================
# 3. SIDEBAR
# =========================================================

st.sidebar.title("📊 AI Business Analyst")

st.sidebar.write(
    "AI-powered business intelligence using "
    "Machine Learning and Generative AI."
)

data_source = st.sidebar.radio(
    "Select Dataset",
    [
        "Sample Dataset",
        "Upload CSV"
    ]
)

uploaded_file = None

if data_source == "Upload CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales dataset",
        type=["csv"]
    )


# =========================================================
# 4. LOAD DATASET
# =========================================================

df = None
source_name = None

try:

    if data_source == "Sample Dataset":

        if not DEFAULT_DATA_PATH.exists():

            st.error(
                f"Sample dataset not found: {DEFAULT_DATA_PATH}"
            )

            st.stop()

        df = pd.read_csv(DEFAULT_DATA_PATH)

        source_name = "sample_data.csv"

    elif data_source == "Upload CSV":

        if uploaded_file is None:

            st.info(
                "Please upload a CSV file to continue."
            )

            st.stop()

        df = pd.read_csv(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        )

        source_name = uploaded_file.name

    df = prepare_dataset(df)

except Exception as error:

    st.error(
        f"Dataset Error: {error}"
    )

    st.stop()


current_source_key = create_source_key(
    df,
    source_name
)


# =========================================================
# 5. HEADER
# =========================================================

st.title("📊 AI Predictive Business Analyst")

st.caption(
    "Sales Forecasting | Demand Forecasting | "
    "Customer Churn | Customer Segmentation | Generative AI"
)

st.divider()

st.write(
    "Analyze business data, predict future trends, "
    "identify customer risks, and ask questions "
    "using an AI Business Analyst."
)


# =========================================================
# 6. RUN COMPLETE ANALYSIS
# =========================================================

if st.sidebar.button(
    "🚀 Run Complete Analysis",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Running Machine Learning models and building Business Context..."
        ):

            business_context = build_business_context(df)

            st.session_state["business_context"] = (
                business_context
            )

            st.session_state["analysis_source_key"] = (
                current_source_key
            )

        st.sidebar.success(
            "Analysis completed successfully!"
        )

    except Exception as error:

        st.error(
            f"Analysis failed: {error}"
        )

        st.stop()


# =========================================================
# 7. CHECK ANALYSIS STATUS
# =========================================================

analysis_ready = (
    "business_context" in st.session_state
    and st.session_state.get("analysis_source_key")
    == current_source_key
)

if not analysis_ready:

    st.warning(
        "Please click 'Run Complete Analysis' "
        "in the sidebar to generate business insights."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.stop()


context = st.session_state["business_context"]

export_data = context["export_data"]


# =========================================================
# 8. DASHBOARD TABS
# =========================================================

tabs = st.tabs([
    "Dataset Overview",
    "Sales Forecast",
    "Product Demand",
    "Customer Churn",
    "Customer Segmentation",
    "AI Business Analyst",
    "Reports & Downloads"
])


# =========================================================
# TAB 1: DATASET OVERVIEW
# =========================================================

with tabs[0]:

    st.header("Dataset Overview")

    overview = context["dataset_overview"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{overview['total_transactions']:,}"
    )

    col2.metric(
        "Unique Customers",
        f"{overview['unique_customers']:,}"
    )

    col3.metric(
        "Unique Products",
        f"{overview['unique_products']:,}"
    )

    col4.metric(
        "Completed Transactions",
        f"{overview['completed_transactions']:,}"
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Dataset Type**")
        st.write(overview["dataset_type"])

        st.write("**Start Date**")
        st.write(overview["date_range"]["start"])

    with col2:

        st.write("**Source**")
        st.write(source_name)

        st.write("**End Date**")
        st.write(overview["date_range"]["end"])

    st.subheader("Transaction Status")

    status_counts = (
        df["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Transactions")
    )

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            status_counts.set_index("Status")
        )

    with col2:

        st.dataframe(
            status_counts,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Monthly Completed Revenue")

    completed_df = df[
        df["Status"].astype(str).str.lower()
        == "completed"
    ].copy()

    monthly_revenue = (
        completed_df
        .groupby(
            completed_df["OrderDate"].dt.to_period("M")
        )["Revenue"]
        .sum()
        .reset_index()
    )

    monthly_revenue.columns = [
        "Month",
        "Revenue"
    ]

    monthly_revenue["Month"] = (
        monthly_revenue["Month"].astype(str)
    )

    st.line_chart(
        monthly_revenue.set_index("Month")
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# TAB 2: SALES FORECASTING
# =========================================================

with tabs[1]:

    st.header("Sales Forecasting")

    sales = context["sales_forecasting"]

    forecast = sales["forecast_summary"]

    evaluation = sales["evaluation_metrics"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Forecast Period",
        f"{sales['forecast_period_days']} Days"
    )

    col2.metric(
        "Predicted Total Revenue",
        f"{forecast['predicted_total_revenue']:,.2f}"
    )

    col3.metric(
        "Average Daily Revenue",
        f"{forecast['average_daily_revenue']:,.2f}"
    )

    col4.metric(
        "Model",
        sales["model"]
    )

    st.subheader("Forecast Summary")

    forecast_table = pd.DataFrame({

        "Metric": [
            "Predicted Total Revenue",
            "Average Daily Revenue",
            "Minimum Daily Revenue",
            "Maximum Daily Revenue"
        ],

        "Value": [
            forecast["predicted_total_revenue"],
            forecast["average_daily_revenue"],
            forecast["minimum_daily_revenue"],
            forecast["maximum_daily_revenue"]
        ]
    })

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Model Evaluation")

    display_evaluation_metrics(evaluation)

    st.caption(
        f"Last historical date: "
        f"{sales['last_historical_date']}"
    )

    st.info(
        "Forecast values are model predictions, "
        "not guaranteed future revenue."
    )


# =========================================================
# TAB 3: PRODUCT DEMAND FORECASTING
# =========================================================

with tabs[2]:

    st.header("Product Demand Forecasting")

    demand = context["product_demand_forecasting"]

    demand_metrics = demand["overall_evaluation_metrics"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Period",
        f"{demand['forecast_period_days']} Days"
    )

    col2.metric(
        "Products Evaluated",
        demand["products_evaluated"]
    )

    col3.metric(
        "Products Skipped",
        demand["products_skipped"]
    )

    st.subheader("Predicted Product Demand")

    product_forecasts = pd.DataFrame(
        demand["top_products_by_predicted_demand"]
    )

    if not product_forecasts.empty:

        st.bar_chart(
            product_forecasts.set_index("Product")[
                "Total_Predicted_Demand"
            ]
        )

        st.dataframe(
            product_forecasts,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Model Evaluation")

    rf_metrics = demand_metrics.get(
        "Random_Forest",
        {}
    )

    baseline_metrics = demand_metrics.get(
        "Seasonal_Naive",
        {}
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Random Forest**")

        display_evaluation_metrics(rf_metrics)

    with col2:

        st.write("**Seasonal Naive Baseline**")

        display_evaluation_metrics(baseline_metrics)

    st.info(
        "Demand predictions estimate future product quantities. "
        "They should be treated as planning estimates."
    )


# =========================================================
# TAB 4: CUSTOMER CHURN PREDICTION
# =========================================================

with tabs[3]:

    st.header("Customer Churn Prediction")

    churn = context["customer_churn_prediction"]

    risk_data = pd.DataFrame(
        churn["risk_distribution"]
    )

    evaluation = churn["evaluation_metrics"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        sum(risk_data["Customer_Count"])
    )

    col2.metric(
        "High Risk Customers",
        int(
            risk_data.loc[
                risk_data["Risk_Level"] == "High Risk",
                "Customer_Count"
            ].sum()
        )
    )

    col3.metric(
        "Churn Observation Window",
        f"{churn['churn_days']} Days"
    )

    st.subheader("Customer Risk Distribution")

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            risk_data.set_index("Risk_Level")[
                "Customer_Count"
            ]
        )

    with col2:

        st.dataframe(
            risk_data,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Model Evaluation")

    evaluation_columns = st.columns(5)

    for i, metric_name in enumerate([
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]):

        value = evaluation.get(metric_name)

        if value is not None:

            evaluation_columns[i].metric(
                metric_name,
                f"{value:.4f}"
            )

    st.subheader("Feature Importance")

    feature_data = pd.DataFrame(
        churn["top_features"]
    )

    if not feature_data.empty:

        st.bar_chart(
            feature_data.set_index("Feature")[
                "Importance"
            ]
        )

    st.write(
        "**Churn Definition:**",
        churn["churn_definition"]
    )

    st.info(
        "Churn risk is a model-estimated probability, "
        "not confirmation that a customer has permanently left."
    )


# =========================================================
# TAB 5: CUSTOMER SEGMENTATION
# =========================================================

with tabs[4]:

    st.header("Customer Segmentation")

    segmentation = context["customer_segmentation"]

    segments = pd.DataFrame(
        segmentation["segment_summary"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        segmentation["total_customers"]
    )

    col2.metric(
        "Number of Segments",
        segmentation["n_clusters"]
    )

    col3.metric(
        "Method",
        segmentation["method"]
    )

    st.subheader("Customer Segments")

    if not segments.empty:

        st.bar_chart(
            segments.set_index("Segment")[
                "Customer_Count"
            ]
        )

        st.dataframe(
            segments,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Business Insights")

    for insight in segmentation["business_insights"]:

        st.markdown(
            f"**{insight['Segment']}** "
            f"({insight['Customer_Count']} customers)"
        )

        st.write(
            insight["Description"]
        )

    st.info(
        "Segment names are interpretations of the "
        "K-Means clustering results."
    )


# =========================================================
# TAB 6: AI BUSINESS ANALYST
# =========================================================

with tabs[5]:

    st.header("🤖 AI Business Analyst")

    st.write(
        "Ask business questions about sales, product demand, "
        "customer churn, and customer segments."
    )

    st.caption(
        "Powered by Ollama and qwen2.5:3b"
    )

    st.divider()

    with st.form("ai_question_form"):

        question = st.text_area(
            "Enter your business question",
            placeholder=(
                "Example: Summarize the sales forecast "
                "and explain the customer churn risk distribution."
            ),
            height=120
        )

        submit_question = st.form_submit_button(
            "Ask AI Business Analyst",
            type="primary"
        )

    if submit_question:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            try:

                with st.spinner(
                    "AI Business Analyst is analyzing the results..."
                ):

                    # Exclude full export DataFrames from AI context.
                    ai_context = get_ai_business_context(
                        context
                    )

                    ai_result = ask_business_question(
                        question=question,
                        business_context=ai_context
                    )

                if ai_result.get("success"):

                    st.subheader("AI Answer")

                    st.markdown(
                        ai_result.get("answer", "")
                    )

                    st.caption(
                        f"Model: {ai_result.get('model', 'Unknown')}"
                    )

                else:

                    st.error(
                        ai_result.get(
                            "error",
                            "AI request failed."
                        )
                    )

            except Exception as error:

                st.error(
                    f"AI Error: {error}"
                )

    st.divider()

    st.caption(
        "The AI assistant explains the available analytical "
        "results. Numerical calculations and ML predictions "
        "are generated by Python and Machine Learning models."
    )


# =========================================================
# TAB 7: REPORTS & DOWNLOADS
# =========================================================

with tabs[6]:

    st.header("📥 Reports & Downloads")

    st.write(
        "Download the complete analytical results "
        "generated by the Machine Learning models."
    )

    st.divider()

    # -----------------------------------------------------
    # SALES FORECAST DOWNLOAD
    # -----------------------------------------------------

    st.subheader("1. Sales Forecast")

    sales_export = export_data["sales_forecast"]

    st.write(
        f"Daily revenue predictions: {len(sales_export):,} rows"
    )

    st.download_button(
        label="Download Sales Forecast CSV",
        data=dataframe_to_csv_bytes(sales_export),
        file_name="sales_forecast.csv",
        mime="text/csv",
        key="download_sales"
    )

    st.divider()

    # -----------------------------------------------------
    # PRODUCT DEMAND DOWNLOAD
    # -----------------------------------------------------

    st.subheader("2. Product Demand Forecast")

    demand_export = export_data["product_demand"]

    st.write(
        f"Product demand predictions: {len(demand_export):,} rows"
    )

    st.download_button(
        label="Download Product Demand CSV",
        data=dataframe_to_csv_bytes(demand_export),
        file_name="product_demand_forecast.csv",
        mime="text/csv",
        key="download_demand"
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER CHURN DOWNLOAD
    # -----------------------------------------------------

    st.subheader("3. Customer Churn Prediction")

    churn_export = export_data["customer_churn"]

    st.write(
        f"Customer predictions: {len(churn_export):,} rows"
    )

    st.download_button(
        label="Download Customer Churn CSV",
        data=dataframe_to_csv_bytes(churn_export),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
        key="download_churn"
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER SEGMENTATION DOWNLOAD
    # -----------------------------------------------------

    st.subheader("4. Customer Segmentation")

    segment_export = export_data["customer_segments"]

    st.write(
        f"Customer segmentation records: "
        f"{len(segment_export):,} rows"
    )

    st.download_button(
        label="Download Customer Segmentation CSV",
        data=dataframe_to_csv_bytes(segment_export),
        file_name="customer_segmentation.csv",
        mime="text/csv",
        key="download_segments"
    )

    st.divider()

    # -----------------------------------------------------
    # BUSINESS SUMMARY TXT DOWNLOAD
    # -----------------------------------------------------

    st.subheader("5. Business Summary Report")

    report_text = build_business_report(context)

    st.write(
        "Download a summary of all business analyses "
        "and model evaluation results."
    )

    st.download_button(
        label="Download Business Summary TXT",
        data=report_text.encode("utf-8-sig"),
        file_name="business_summary_report.txt",
        mime="text/plain",
        key="download_business_report"
    )

    st.divider()

    st.success(
        "All export files are ready."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Predictive Business Analyst | "
    "Basic Version | Machine Learning + Generative AI"
)
