
import pandas as pd

from agents.orchestrator import run_analytics_workflow


df = pd.DataFrame({
    "Product": ["Laptop", "Mobile", "Tablet", "Laptop", "Mobile"],
    "Category": [
        "Electronics", "Electronics", "Electronics",
        "Electronics", "Electronics"
    ],
    "Sales": [50000, 20000, 15000, 55000, 25000],
    "Quantity": [2, 3, 1, 2, 4]
})


try:

    print("Starting Analytics Workflow...")

    result = run_analytics_workflow(
        df,
        cleaning_approved=True
    )

    print("\nWORKFLOW COMPLETED!")

    print("\n1. DATA UNDERSTANDING")
    print(result.get("understanding_report"))

    print("\n2. DATA CLEANING")
    print(result.get("cleaning_report"))

    print("\n3. VISUALIZATION")
    print(result.get("visualization_report"))

    print("\n4. AI INSIGHTS")
    print(result.get("ai_insight"))

    print("\n5. PDF REPORT")
    print(result.get("report_path"))

    print("\nCURRENT STEP:")
    print(result.get("current_step"))

except Exception as error:

    print("\nWORKFLOW FAILED!")
    print(type(error).__name__, error)

    raise

