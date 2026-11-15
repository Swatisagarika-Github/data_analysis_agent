from typing import TypedDict, Optional, Any
import pandas as pd


class AnalyticsState(TypedDict, total=False):
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame

    understanding_report: dict
    cleaning_report: Any

    cleaning_approved: bool

    eda_report: dict
    visualization_report: Any

    ai_insight: str
    report_path: str

    current_step: str
    error: Optional[str]