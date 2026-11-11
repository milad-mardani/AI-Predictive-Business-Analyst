
from modules.ai_business_analyst import (
    check_ollama_connection,
    ask_business_question
)


# =========================================================
# Test 1: Ollama Connection
# =========================================================

print("\n========== OLLAMA CONNECTION ==========")

connection = check_ollama_connection()

print(connection)


if not connection["success"]:
    raise SystemExit("Ollama connection failed.")


# =========================================================
# Test 2: Sample Business Context
# =========================================================

sample_context = {

    "dataset": "Synthetic E-commerce Dataset",

    "sales_forecasting": {

        "forecast_period": "Next 30 days",

        "predicted_total_revenue": 263830.52,

        "average_daily_revenue": 8794.35,

        "historical_evaluation": {

            "model": "Random Forest",

            "WAPE": 41.94

        }

    },

    "customer_churn": {

        "high_risk_customers": 174,

        "medium_risk_customers": 82,

        "low_risk_customers": 244

    }

}


# =========================================================
# Test 3: Ask Business Question
# =========================================================

print("\n========== AI BUSINESS ANALYST ==========")

question = (
    "Summarize the sales forecast and explain "
    "the customer churn risk distribution."
)

result = ask_business_question(
    question=question,
    business_context=sample_context
)


if result["success"]:

    print("\nQuestion:")
    print(result["question"])

    print("\nAI Answer:")
    print(result["answer"])

    print("\nModel:")
    print(result["model"])

else:

    print("\nError:")
    print(result["error"])
