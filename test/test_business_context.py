
import json
from pathlib import Path

import pandas as pd

from modules.business_context import build_business_context
from modules.ai_business_analyst import ask_business_question


# =========================================================
# 1. Load Dataset
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "sample_data.csv"

print("\n========== LOADING DATASET ==========")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows:", len(df))


# =========================================================
# 2. Build Business Context
# =========================================================

print("\n========== BUILDING BUSINESS CONTEXT ==========")

business_context = build_business_context(df)


# =========================================================
# 3. Display Business Context
# =========================================================

print("\n========== BUSINESS CONTEXT ==========")

print(
    json.dumps(
        business_context,
        indent=2,
        ensure_ascii=False,
        default=str
    )
)


# =========================================================
# 4. Ask AI Business Question
# =========================================================

print("\n========== AI BUSINESS ANALYST ==========")

question = """
Analyze the business using the available results.

Summarize:
1. The sales forecast.
2. The product demand forecast.
3. Customer churn risk.
4. Customer segments.

Explain the important findings and provide practical
business recommendations.

Do not invent numerical values.
"""

result = ask_business_question(
    question=question,
    business_context=business_context
)


# =========================================================
# 5. Display AI Answer
# =========================================================

if result["success"]:

    print("\nQuestion:")
    print(result["question"])

    print("\nAI Business Analyst Answer:")
    print(result["answer"])

    print("\nModel:")
    print(result["model"])

else:

    print("\nAI Error:")
    print(result["error"])
