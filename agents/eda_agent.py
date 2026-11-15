
from module.statistics import numerical_analysis

from module.eda import (
    categorical_analysis,
    correlation_analysis,
    category_comparison,
    time_trend_analysis
)


def run_eda_agent(df):

    report = {}

    # Numerical Analysis
    report["numerical_analysis"] = numerical_analysis(df)

    # Categorical Analysis
    report["categorical_analysis"] = categorical_analysis(df)

    # Correlation Analysis
    report["correlation"] = correlation_analysis(df)

    # Category-wise Comparison
    report["category_comparison"] = category_comparison(df)

    # Date Trend Analysis
    report["time_trends"] = time_trend_analysis(df)

    return report
