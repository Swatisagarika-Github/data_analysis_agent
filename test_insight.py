
from agents.insight_agent import run_insight_agent

sample_eda_report = {
    "numerical_analysis": {
        "Revenue": {
            "count": 3,
            "mean": 123333.33,
            "median": 120000,
            "minimum": 100000,
            "maximum": 150000
        }
    },

    "categorical_analysis": {
        "Month": {
            "unique_categories": 3,
            "most_common": "January"
        }
    },

    "revenue_by_month": {
        "January": 100000,
        "February": 150000,
        "March": 120000
    }
}

try:
    result = run_insight_agent(sample_eda_report)

    print("\n========== AI GENERATED INSIGHTS ==========\n")
    print(result)

except Exception as error:
    print("Error:", error)
