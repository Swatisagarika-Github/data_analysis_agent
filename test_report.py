
import pandas as pd

from agents.report_agent import run_report_agent


# Sample dataset
df = pd.DataFrame({
    "Product": ["Laptop", "Mobile", "Tablet", "Laptop", "Mobile"],
    "Category": ["Electronics", "Electronics", "Electronics",
                 "Electronics", "Electronics"],
    "Sales": [50000, 30000, 20000, 45000, 35000],
    "Quantity": [5, 10, 7, 4, 8]
})

# Sample EDA results
eda_report = {
    "numerical_analysis": {
        "Sales": {
            "mean": 36000,
            "median": 35000
        }
    }
}

# Sample AI insight
ai_insight = """
Dataset Overview:
The dataset contains five product records.

Key Findings:
Laptop sales have the highest individual sales values.

Recommendations:
Monitor product sales and inventory regularly.
"""

try:
    report_path = run_report_agent(
        df=df,
        eda_report=eda_report,
        ai_insight=ai_insight
    )

    print("PDF report generated successfully!")
    print("Report location:", report_path)

except Exception as error:
    print("Report generation failed:", error)
