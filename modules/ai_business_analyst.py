
import json
import re
from datetime import datetime

import ollama


# =========================================================
# 1. Configuration
# =========================================================

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_CTX = 2048
DEFAULT_NUM_PREDICT = 200

MAX_CONTEXT_LENGTH = 20000


# =========================================================
# 2. System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are an AI Business Analyst.

Your role is to explain business data and machine learning results
using the information provided by Python and the analytical modules.

STRICT RULES:

1. Use ONLY the information provided in BUSINESS CONTEXT.

2. Python and Machine Learning models are responsible for calculations
   and predictions. You must not calculate or invent numerical results.

3. Never invent revenue, sales, percentages, customer counts,
   forecasts, model metrics, or other statistics.

4. Never claim that a particular factor caused a business result
   unless the provided context explicitly confirms that relationship.

5. If the available information is insufficient to answer a question,
   clearly explain what information is missing.

6. Distinguish historical results from model predictions.

7. Explain machine learning results carefully.
   Do not describe predictions as guaranteed future outcomes.

8. Recommendations must be clearly identified as suggestions,
   not verified findings.

9. Do not invent information about the dataset.

10. Answer in the same language as the user's question.

11. Never output placeholders.

12. Extract numerical values exactly as they appear in the context.
    Do not replace available values with placeholders.

Keep answers clear, useful, and concise.
"""


# =========================================================
# 3. Ollama Connection
# =========================================================

def check_ollama_connection():

    try:

        response = ollama.list()

        available_models = [
            model.model for model in response.models
        ]

        return {
            "success": True,
            "models": available_models
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


def check_model_availability(model_name=DEFAULT_MODEL):

    connection = check_ollama_connection()

    if not connection["success"]:
        return False

    return model_name in connection["models"]


# =========================================================
# 4. Prepare Business Context
# =========================================================

def prepare_business_context(business_context):

    if business_context is None:
        raise ValueError("Business context cannot be empty.")

    if isinstance(business_context, str):

        context_text = business_context

    else:

        context_text = json.dumps(
            business_context,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    if not context_text.strip():

        raise ValueError("Business context cannot be empty.")

    if len(context_text) > MAX_CONTEXT_LENGTH:

        raise ValueError(
            "Business context is too large. "
            "Please provide a smaller context."
        )

    return context_text


# =========================================================
# 5. Find Nested Context
# =========================================================

def find_context_section(data, required_keys):

    """
    Find a dictionary containing all required keys
    inside a nested business context.
    """

    if isinstance(data, str):

        try:
            data = json.loads(data)

        except (json.JSONDecodeError, TypeError):
            return None

    if isinstance(data, dict):

        if all(key in data for key in required_keys):
            return data

        for value in data.values():

            result = find_context_section(
                value,
                required_keys
            )

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_context_section(
                item,
                required_keys
            )

            if result is not None:
                return result

    return None


# =========================================================
# 6. Find Sales Forecast Summary
# =========================================================

def find_forecast_summary(data):

    """
    Find the sales forecast summary in nested business context.
    """

    if isinstance(data, str):

        try:
            data = json.loads(data)

        except (json.JSONDecodeError, TypeError):
            return None

    if isinstance(data, dict):

        summary = data.get("forecast_summary")

        if isinstance(summary, dict):

            if (
                "predicted_total_revenue" in summary
                or "daily_forecast" in summary
            ):
                return summary

        for value in data.values():

            result = find_forecast_summary(value)

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_forecast_summary(item)

            if result is not None:
                return result

    return None


# =========================================================
# 7. Extract Requested Dates
# =========================================================

def extract_requested_dates(question):

    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b",
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b"
    ]

    found_dates = []

    for pattern in date_patterns:

        matches = re.findall(
            pattern,
            question,
            flags=re.IGNORECASE
        )

        for value in matches:

            parsed_date = None

            for date_format in [
                "%Y-%m-%d",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y"
            ]:

                try:

                    parsed_date = datetime.strptime(
                        value,
                        date_format
                    ).date()

                    break

                except ValueError:
                    continue

            if parsed_date is not None:

                date_string = parsed_date.isoformat()

                if date_string not in found_dates:
                    found_dates.append(date_string)

    return found_dates


# =========================================================
# 8. Exact Product Demand Extraction
# =========================================================

def answer_product_demand_question(
    question,
    business_context
):

    """
    Extract product demand forecast values directly
    from Python-generated business context.
    """

    demand_context = find_context_section(
        business_context,
        [
            "top_products_by_predicted_demand"
        ]
    )

    if demand_context is None:
        return None

    question_lower = question.lower()

    has_product_reference = any(
        keyword in question_lower
        for keyword in [
            "product",
            "products",
            "محصول",
            "محصولات"
        ]
    )

    has_demand_reference = any(
        keyword in question_lower
        for keyword in [
            "demand",
            "predicted demand",
            "forecast demand",
            "تقاضا",
            "میزان تقاضا"
        ]
    )

    has_top_reference = any(
        keyword in question_lower
        for keyword in [
            "top",
            "highest",
            "most",
            "greatest",
            "معرفی",
            "بیشترین",
            "بالاترین"
        ]
    )

    if not (
        has_product_reference
        and has_demand_reference
        and has_top_reference
    ):
        return None

    # Detect the requested number of products.
    number_match = re.search(
        r"\b(?:top|first)\s+(\d+)\b",
        question_lower
    )

    if number_match:

        requested_count = int(
            number_match.group(1)
        )

    else:

        number_match = re.search(
            r"\b(\d+)\s+products?\b",
            question_lower
        )

        requested_count = (
            int(number_match.group(1))
            if number_match
            else 5
        )

    if requested_count <= 0:
        return None

    products = demand_context.get(
        "top_products_by_predicted_demand",
        []
    )

    if not isinstance(products, list) or not products:

        return (
            "Product demand forecast results are not available "
            "in the business context."
        )

    forecast_days = demand_context.get(
        "forecast_period_days",
        30
    )

    answers = [
        f"Top {min(requested_count, len(products))} Products "
        f"by Predicted Demand (Next {forecast_days} Days):"
    ]

    selected_products = products[:requested_count]

    for index, product in enumerate(
        selected_products,
        start=1
    ):

        if not isinstance(product, dict):
            continue

        product_name = product.get(
            "Product",
            "Unknown"
        )

        predicted_demand = product.get(
            "Total_Predicted_Demand"
        )

        if predicted_demand is None:

            answers.append(
                f"{index}. {product_name} | "
                "Predicted demand: Not available"
            )

        else:

            answers.append(
                f"{index}. {product_name} | "
                f"Total Predicted Demand: {predicted_demand}"
            )

    if len(answers) == 1:
        return None

    return "\n".join(answers)


# =========================================================
# 9. Exact Sales Forecast Extraction
# =========================================================

def answer_sales_forecast_question(
    question,
    business_context
):

    summary = find_forecast_summary(
        business_context
    )

    if summary is None:
        return None

    question_lower = question.lower()

    has_forecast_reference = any(
        keyword in question_lower
        for keyword in [
            "forecast",
            "predicted",
            "prediction",
            "پیش بینی",
            "پیش‌بینی"
        ]
    )

    has_sales_reference = any(
        keyword in question_lower
        for keyword in [
            "revenue",
            "sales",
            "درآمد",
            "فروش"
        ]
    )

    # Prevent product demand questions from entering
    # the sales forecast extraction route.
    if not (
        has_forecast_reference
        and has_sales_reference
    ):
        return None

    total_requested = (

        (
            "total" in question_lower
            and "revenue" in question_lower
        )

        or "next 30 days" in question_lower

        or "30-day revenue" in question_lower

        or "total predicted revenue" in question_lower

        or "درآمد کل" in question

        or "مجموع درآمد" in question
    )

    requested_dates = extract_requested_dates(
        question
    )

    daily_forecast = summary.get(
        "daily_forecast",
        []
    )

    forecast_lookup = {

        str(item.get("date")): item.get(
            "predicted_revenue"
        )

        for item in daily_forecast

        if isinstance(item, dict)
    }

    answers = []

    if total_requested:

        total_revenue = summary.get(
            "predicted_total_revenue"
        )

        if total_revenue is not None:

            answers.append(
                "Total predicted revenue for the next "
                f"{summary.get('forecast_period_days', 30)} days: "
                f"{float(total_revenue):,.2f}"
            )

        else:

            answers.append(
                "Total predicted revenue: "
                "Not available in the business context."
            )

    for requested_date in requested_dates:

        value = forecast_lookup.get(
            requested_date
        )

        if value is not None:

            answers.append(
                f"Predicted revenue on {requested_date}: "
                f"{float(value):,.2f}"
            )

        else:

            answers.append(
                f"Predicted revenue on {requested_date}: "
                "Not available in the business context."
            )

    if not answers:
        return None

    return "\n".join(answers)


# =========================================================
# 10. Exact Customer Churn Extraction
# =========================================================

def answer_customer_churn_question(
    question,
    business_context
):

    churn_context = find_context_section(
        business_context,
        [
            "high_risk_customer_count",
            "high_risk_customer_percentage",
            "top_5_customers_by_churn_probability"
        ]
    )

    if churn_context is None:
        return None

    question_lower = question.lower()

    has_churn_reference = any(
        keyword in question_lower
        for keyword in [
            "churn",
            "customer risk",
            "high risk",
            "medium risk",
            "low risk",
            "ریزش مشتری",
            "ریسک مشتری",
            "پرریسک"
        ]
    )

    if not has_churn_reference:
        return None

    answers = []

    count_requested = any(
        keyword in question_lower
        for keyword in [
            "how many",
            "number of customers",
            "customer count",
            "high risk customers",
            "تعداد",
            "چند مشتری"
        ]
    )

    if count_requested:

        count = churn_context.get(
            "high_risk_customer_count"
        )

        if count is not None:

            answers.append(
                f"Number of High Risk customers: {count}"
            )

        else:

            answers.append(
                "High Risk customer count: Not available."
            )

    percentage_requested = any(
        keyword in question_lower
        for keyword in [
            "percentage",
            "percent",
            "proportion",
            "high risk rate",
            "درصد"
        ]
    )

    if percentage_requested:

        percentage = churn_context.get(
            "high_risk_customer_percentage"
        )

        if percentage is not None:

            answers.append(
                f"Percentage of High Risk customers: "
                f"{percentage}%"
            )

        else:

            answers.append(
                "High Risk customer percentage: Not available."
            )

    top_requested = any(
        keyword in question_lower
        for keyword in [
            "top",
            "highest",
            "most likely",
            "highest churn risk",
            "highest risk",
            "list the customers",
            "معرفی مشتری",
            "بالاترین ریسک",
            "پنج مشتری",
            "۵ مشتری"
        ]
    )

    if top_requested:

        top_customers = churn_context.get(
            "top_5_customers_by_churn_probability",
            []
        )

        if top_customers:

            answers.append(
                "\nTop 5 customers by churn probability:"
            )

            for index, customer in enumerate(
                top_customers,
                start=1
            ):

                customer_id = customer.get(
                    "CustomerID",
                    "Unknown"
                )

                probability = customer.get(
                    "Churn_Probability"
                )

                risk_level = customer.get(
                    "Risk_Level",
                    "Unknown"
                )

                if probability is None:

                    answers.append(
                        f"{index}. {customer_id} | "
                        f"Risk: {risk_level}"
                    )

                else:

                    answers.append(
                        f"{index}. {customer_id} | "
                        f"Churn Probability: {probability} | "
                        f"Risk: {risk_level}"
                    )

        else:

            answers.append(
                "Top-risk customer records are not available."
            )

    if not answers:
        return None

    return "\n".join(answers)


# =========================================================
# 11. Exact Customer Segmentation Extraction
# =========================================================

def answer_customer_segmentation_question(
    question,
    business_context
):

    """
    Extract customer segment counts and percentages directly
    from Python-generated segmentation context.
    """

    segmentation_context = find_context_section(
        business_context,
        [
            "segment_summary"
        ]
    )

    if segmentation_context is None:
        return None

    question_lower = question.lower()

    has_segmentation_reference = any(
        keyword in question_lower
        for keyword in [
            "segmentation",
            "customer segment",
            "customer segments",
            "segments",
            "segment summary",
            "بخش بندی",
            "بخش‌بندی",
            "گروه مشتری",
            "گروه‌های مشتری"
        ]
    )

    if not has_segmentation_reference:
        return None

    count_requested = any(
        keyword in question_lower
        for keyword in [
            "how many",
            "number of customers",
            "customer count",
            "count",
            "تعداد",
            "چند مشتری"
        ]
    )

    percentage_requested = any(
        keyword in question_lower
        for keyword in [
            "percentage",
            "percent",
            "proportion",
            "درصد"
        ]
    )

    if not count_requested and not percentage_requested:
        return None

    segment_summary = segmentation_context.get(
        "segment_summary",
        []
    )

    if not isinstance(segment_summary, list) or not segment_summary:
        return (
            "Customer segment summary is not available "
            "in the business context."
        )

    answers = [
        "Customer Segmentation Results:"
    ]

    for segment in segment_summary:

        if not isinstance(segment, dict):
            continue

        segment_name = segment.get(
            "Segment",
            "Unknown"
        )

        customer_count = segment.get(
            "Customer_Count"
        )

        customer_percentage = segment.get(
            "Customer_Percent"
        )

        parts = [str(segment_name)]

        if count_requested:

            if customer_count is not None:
                parts.append(
                    f"{customer_count} customers"
                )
            else:
                parts.append(
                    "Customer count: Not available"
                )

        if percentage_requested:

            if customer_percentage is not None:

                parts.append(
                    f"{float(customer_percentage):.1f}%"
                )

            else:

                parts.append(
                    "Percentage: Not available"
                )

        answers.append(
            " - " + " | ".join(parts)
        )

    if len(answers) == 1:
        return None

    return "\n".join(answers)


# =========================================================
# 12. Ask Business Question
# =========================================================

def ask_business_question(
    question,
    business_context,
    model_name=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE
):

    if not isinstance(question, str) or not question.strip():

        return {
            "success": False,
            "error": "Please enter a valid business question."
        }

    try:

        # Step 1: Exact Product Demand Extraction

        exact_answer = answer_product_demand_question(
            question,
            business_context
        )

        if exact_answer is not None:

            return {
                "success": True,
                "question": question,
                "answer": exact_answer,
                "model": "Python - Business Context Extraction"
            }

        # Step 2: Exact Sales Forecast Extraction

        exact_answer = answer_sales_forecast_question(
            question,
            business_context
        )

        if exact_answer is not None:

            return {
                "success": True,
                "question": question,
                "answer": exact_answer,
                "model": "Python - Business Context Extraction"
            }

        # Step 3: Exact Customer Churn Extraction

        exact_answer = answer_customer_churn_question(
            question,
            business_context
        )

        if exact_answer is not None:

            return {
                "success": True,
                "question": question,
                "answer": exact_answer,
                "model": "Python - Business Context Extraction"
            }

        # Step 4: Exact Customer Segmentation Extraction

        exact_answer = answer_customer_segmentation_question(
            question,
            business_context
        )

        if exact_answer is not None:

            return {
                "success": True,
                "question": question,
                "answer": exact_answer,
                "model": "Python - Business Context Extraction"
            }

        # Step 5: Prepare Context for Ollama

        context_text = prepare_business_context(
            business_context
        )

        # Step 6: Check Model Availability

        if not check_model_availability(model_name):

            return {
                "success": False,
                "error": (
                    f"Model '{model_name}' is not available. "
                    "Check your Ollama installation."
                )
            }

        # Step 7: Build Prompt

        user_prompt = f"""
BUSINESS CONTEXT:
{context_text}

BUSINESS QUESTION:
{question}

IMPORTANT:
Answer using only the provided business context.
Do not invent or calculate numerical values.
Use exact available values.
If information is missing, explicitly say so.
Never output placeholder variables.
"""

        # Step 8: Ollama

        response = ollama.chat(

            model=model_name,

            messages=[

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            options={

                "temperature": temperature,

                "num_ctx": DEFAULT_NUM_CTX,

                "num_predict": DEFAULT_NUM_PREDICT

            }
        )

        answer = response.message.content.strip()

        if not answer:

            return {
                "success": False,
                "error": "The AI model returned an empty response."
            }

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "model": model_name
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }
