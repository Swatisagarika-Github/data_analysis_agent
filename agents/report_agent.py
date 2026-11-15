from module.report_generator import generate_pdf_report
from report_generator_with_engine import generate_pdf_report

def run_report_agent(
    df,
    eda_report,
    ai_insight=None,
    cleaning_report=None,
    output_path="reports/analysis_report.pdf"
):
    """
    Generate PDF report using dataset,
    EDA results, cleaning information and AI insights.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty. Cannot generate PDF.")

    if not isinstance(eda_report, dict):
        raise ValueError("Invalid EDA report.")

    report_path = generate_pdf_report(
        df=df,
        eda_report=eda_report,
        ai_insight=ai_insight,
        cleaning_report=cleaning_report,
        output_path=output_path
    )

    return report_path