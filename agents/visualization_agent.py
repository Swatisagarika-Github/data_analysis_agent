
import pandas as pd


def run_visualization_agent(df):
    recommendations = []

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    date_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    # Category + numerical value → Bar chart
    for category in categorical_columns:
        for numerical in numerical_columns:
            recommendations.append({
                "chart": "Bar Chart",
                "x": category,
                "y": numerical,
                "reason": f"Compare {numerical} across {category}."
            })

    # Date + numerical value → Line chart
    for date in date_columns:
        for numerical in numerical_columns:
            recommendations.append({
                "chart": "Line Chart",
                "x": date,
                "y": numerical,
                "reason": f"Show {numerical} trends over time."
            })

    # One numerical column → Histogram
    for numerical in numerical_columns:
        recommendations.append({
            "chart": "Histogram",
            "x": numerical,
            "y": None,
            "reason": f"Understand the distribution of {numerical}."
        })

    # Two numerical columns → Scatter plot
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            recommendations.append({
                "chart": "Scatter Plot",
                "x": numerical_columns[i],
                "y": numerical_columns[j],
                "reason": (
                    f"Explore the relationship between "
                    f"{numerical_columns[i]} and {numerical_columns[j]}."
                )
            })

    # Category proportions → Pie chart
    for category in categorical_columns:
        if df[category].nunique(dropna=True) <= 8:
            recommendations.append({
                "chart": "Pie Chart",
                "x": category,
                "y": None,
                "reason": f"Show the proportions of {category}."
            })

    if len(numerical_columns) >= 2:
        recommendations.append({
            "chart": "Correlation Heatmap",
            "x": None,
            "y": None,
            "reason": "Explore correlations between numerical columns."
        })

    return recommendations
