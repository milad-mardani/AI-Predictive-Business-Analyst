# AI-Powered Predictive Business Analyst

An AI-powered business analytics application that combines **Machine Learning, Generative AI, and business intelligence** to analyze sales data, forecast future performance, predict customer churn, and generate business insights.

The application uses Python and Scikit-learn for data analysis and predictive modeling, Streamlit for the user interface, and a locally running Ollama LLM for natural-language business analysis.

## Key Features

### 1. Sales Revenue Forecasting
- Forecasts sales revenue for the next 30 days.
- Uses a Random Forest Regressor.
- Evaluates model performance using MAE, RMSE, and WAPE.
- Uses historical completed-order revenue for forecasting.

### 2. Product Demand Forecasting
- Predicts the next 30 days of demand for individual products.
- Uses Random Forest models.
- Compares model performance with a Seasonal Naive baseline.
- Reports forecasting evaluation metrics.

### 3. Customer Churn Prediction
- Identifies customers with different levels of churn risk.
- Defines churn as having no completed purchase during the following 90 days.
- Uses customer behavior features and a Random Forest classifier.
- Evaluates the model using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### 4. Customer Segmentation
- Applies RFM analysis (Recency, Frequency, and Monetary).
- Uses K-Means clustering to identify four customer segments.
- Provides business-oriented segment labels:
  - Champions
  - Loyal Customers
  - Potential Loyalists
  - At Risk

### 5. AI Business Analyst
- Allows users to ask business questions in natural language.
- Uses Ollama with the Qwen 2.5:3b model.
- Generates explanations based on verified analytical and machine learning results.
- Distinguishes historical results from model predictions.

Python performs the numerical calculations and prepares the business context. The LLM is used to interpret and explain the available results rather than independently calculate business metrics.

### 6. Interactive Dashboard and Reports
- Interactive Streamlit dashboard.
- Dataset overview and business performance analysis.
- Dedicated tabs for forecasting, churn prediction, segmentation, and AI-powered analysis.
- Downloadable CSV reports and a business summary report.

## Project Architecture

```text
Sales Dataset
     |
     v
Data Validation
     |
     v
Business Context Builder
     |
     +-------------------------+
     |           |             |
     v           v             v
Sales       Demand          Customer
Forecasting Forecasting     Churn Prediction
     |           |             |
     +-----------+-------------+
                 |
                 v
        Customer Segmentation
                 |
                 v
        AI Business Analyst
                 |
                 v
          Ollama / Qwen
                 |
                 v
        Business Insights
                 |
                 v
        Streamlit Dashboard
                 |
                 v
         Downloadable Reports
```

## Project Structure

```text
AI-Predictive-Business-Analyst/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sample_data.csv
│
├── modules/
│   ├── ai_business_analyst.py
│   ├── business_context.py
│   ├── churn_prediction.py
│   ├── customer_segmentation.py
│   ├── data_analysis.py
│   ├── data_processing.py
│   ├── demand_forecasting.py
│   └── sales_forecasting.py
│
└── test/
    ├── test_ai_business_analyst.py
    └── test_business_context.py
```

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Ollama**
- **Qwen 2.5:3b**
- **Matplotlib**
- **Machine Learning**
- **Generative AI**

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Predictive-Business-Analyst.git
cd AI-Predictive-Business-Analyst
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

Download and install Ollama:

https://ollama.com/

Pull the required model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before using the AI Business Analyst feature.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Dataset

The project includes a sample sales dataset:

`data/sample_data.csv`

The application also supports uploading a custom CSV dataset through the Streamlit interface.

Required columns:

- OrderID
- OrderDate
- CustomerID
- Product
- Category
- Quantity
- UnitPrice
- Revenue
- Region
- Status

The sample dataset is synthetic and is provided for demonstration and testing purposes.

## Testing

Two manual test scripts are included in the `test/` directory:

- `test_ai_business_analyst.py`: Checks the AI Business Analyst functionality.
- `test_business_context.py`: Checks the business context generation and AI analysis workflow.

These are manual test scripts, not a fully automated pytest test suite.

## Important Notes and Limitations

- Forecasts are estimates and are not guaranteed future outcomes.
- Model performance depends on the quality and history of the supplied dataset.
- Churn is defined using a 90-day period without a completed purchase.
- Customer segment labels are business interpretations of K-Means clusters.
- The sample dataset is synthetic; real-world performance should be evaluated using actual business data.
- The AI Business Analyst depends on the local Ollama model and the analytical context provided by the application.

## Project Goal

This project demonstrates how traditional business analytics and predictive machine learning can be combined with Generative AI to build an interactive business analysis assistant.

It showcases practical skills in data processing, predictive modeling, customer analytics, LLM integration, and AI-powered business intelligence.

---

**Project Type:** Portfolio Project  
**Focus:** Machine Learning + Generative AI + Business Analytics  
**Application:** Streamlit  
**LLM:** Ollama / Qwen 2.5:3b
