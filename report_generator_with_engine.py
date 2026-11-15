
import os
import math
import textwrap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

from module.algorithm_recommender import analyze_dataset
from module.model_evaluator import evaluate_models


# ==========================================================
# 1. HELPER FUNCTIONS
# ==========================================================

def safe_value(value):

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass

    if isinstance(value, float):

        if math.isinf(value) or math.isnan(value):
            return "N/A"

        return round(value, 2)

    return value


def safe_percentage(value, total):

    if total == 0:
        return 0

    return round((value / total) * 100, 2)


def make_paragraph(value, style):

    return Paragraph(
        escape(str(safe_value(value))),
        style
    )


def create_report_table(
    data,
    styles,
    col_widths=None,
    font_size=8
):

    if not data:
        return Paragraph("No data available.", styles["Normal"])

    formatted_data = []

    for row_index, row in enumerate(data):

        formatted_row = []

        for value in row:

            if row_index == 0:
                style = styles["TableHeader"]
            else:
                style = styles["TableCell"]

            formatted_row.append(
                Paragraph(
                    escape(str(safe_value(value))),
                    style
                )
            )

        formatted_data.append(formatted_row)

    table = Table(
        formatted_data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("ALIGN", (0, 0), (-1, -1), "LEFT"),

        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),

        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F2F6FA")
        ])

    ]))

    return table


# ==========================================================
# 2. DATA QUALITY ANALYSIS
# ==========================================================

def create_data_quality(df):

    total_rows = len(df)
    total_columns = len(df.columns)

    missing_values = int(df.isnull().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    complete_rows = int(
        df.dropna().shape[0]
    )

    missing_percentage = safe_percentage(
        missing_values,
        total_rows * total_columns
    )

    rows = [

        ["Metric", "Result"],

        ["Total Rows", total_rows],

        ["Total Columns", total_columns],

        ["Missing Values", missing_values],

        ["Duplicate Rows", duplicate_rows],

        ["Complete Rows", complete_rows],

        ["Missing Values Percentage", f"{missing_percentage}%"]

    ]

    return rows


# ==========================================================
# 3. STATISTICAL ANALYSIS
# ==========================================================

def create_metric_table(df):

    numeric_df = df.select_dtypes(include="number")

    rows = [[
        "Column",
        "Mean",
        "Median",
        "Minimum",
        "Maximum"
    ]]

    for column in numeric_df.columns:

        rows.append([

            column,

            safe_value(numeric_df[column].mean()),

            safe_value(numeric_df[column].median()),

            safe_value(numeric_df[column].min()),

            safe_value(numeric_df[column].max())

        ])

    return rows


# ==========================================================
# 4. VISUALIZATION GENERATION
# ==========================================================

def create_charts(df, chart_folder):

    os.makedirs(chart_folder, exist_ok=True)

    chart_paths = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical histograms

    for column in numeric_columns[:5]:

        try:

            plt.figure(figsize=(7, 4))

            df[column].dropna().hist(bins=20)

            plt.title(f"Distribution of {column}")

            plt.xlabel(column)

            plt.ylabel("Frequency")

            plt.tight_layout()

            path = os.path.join(
                chart_folder,
                f"histogram_{len(chart_paths)}.png"
            )

            plt.savefig(path, dpi=120)

            plt.close()

            chart_paths.append(path)

        except Exception:
            plt.close()
            continue

    # Correlation heatmap

    if len(numeric_columns) >= 2:

        try:

            correlation = df[numeric_columns].corr()

            plt.figure(figsize=(8, 6))

            plt.imshow(
                correlation,
                cmap="coolwarm",
                aspect="auto",
                vmin=-1,
                vmax=1
            )

            plt.colorbar()

            plt.xticks(
                range(len(correlation.columns)),
                correlation.columns,
                rotation=45,
                ha="right"
            )

            plt.yticks(
                range(len(correlation.columns)),
                correlation.columns
            )

            plt.title("Correlation Heatmap")

            plt.tight_layout()

            path = os.path.join(
                chart_folder,
                "correlation_heatmap.png"
            )

            plt.savefig(path, dpi=120)

            plt.close()

            chart_paths.append(path)

        except Exception:
            plt.close()

    # Categorical bar charts

    for column in categorical_columns[:5]:

        try:

            counts = df[column].value_counts().head(10)

            if counts.empty:
                continue

            plt.figure(figsize=(7, 4))

            counts.plot(kind="bar")

            plt.title(f"Category Distribution: {column}")

            plt.xlabel(column)

            plt.ylabel("Count")

            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()

            path = os.path.join(
                chart_folder,
                f"category_{len(chart_paths)}.png"
            )

            plt.savefig(path, dpi=120)

            plt.close()

            chart_paths.append(path)

        except Exception:
            plt.close()
            continue

    return chart_paths


# ==========================================================
# 5. IMPLEMENTED TECHNIQUES
# ==========================================================

def get_implemented_techniques(
    df,
    cleaning_report=None
):

    techniques = []

    techniques.append("Dataset profiling")

    techniques.append("Numerical and categorical column identification")

    if df.isnull().sum().sum() > 0:
        techniques.append("Missing value detection")

    if df.duplicated().sum() > 0:
        techniques.append("Duplicate record detection")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        techniques.append("Numerical statistical analysis")

        techniques.append("IQR-based outlier detection")

        techniques.append("Histogram-based distribution analysis")

    if len(numeric_columns) >= 2:

        techniques.append("Correlation analysis")

        techniques.append("Correlation heatmap generation")

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if categorical_columns:

        techniques.append("Categorical frequency analysis")

        techniques.append("Categorical distribution visualization")

    if cleaning_report:

        if isinstance(cleaning_report, dict):

            for key, value in cleaning_report.items():

                if value:
                    techniques.append(str(key))

        elif isinstance(cleaning_report, list):

            techniques.extend(
                str(item) for item in cleaning_report
            )

        else:

            techniques.append(str(cleaning_report))

    return list(dict.fromkeys(techniques))


# ==========================================================
# 6. LEGACY ALGORITHM SUGGESTIONS
# ==========================================================

def recommend_ml_algorithms(df):

    recommendations = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if numeric_columns:

        recommendations.extend([

            "Linear Regression",

            "Decision Tree",

            "Random Forest"

        ])

    if categorical_columns:

        recommendations.extend([

            "Logistic Regression",

            "Decision Tree Classifier",

            "Random Forest Classifier"

        ])

    return list(dict.fromkeys(recommendations))


# ==========================================================
# 7. ALGORITHM RECOMMENDATION ENGINE
# ==========================================================

def create_algorithm_engine_section(df):

    result = analyze_dataset(df)

    summary_rows = [[
        "Parameter",
        "Result"
    ]]

    dataset_info = result.get(
        "dataset_information",
        {}
    )

    if not isinstance(dataset_info, dict):
        dataset_info = {}

    summary_rows.extend([

        ["Dataset Information", str(dataset_info)],

        [
            "Selected Target",
            result.get("selected_target") or "Not selected"
        ],

        [
            "Detected Task",
            result.get("detected_task") or "Not identified"
        ]

    ])

    # Target candidates

    candidate_rows = [[

        "Rank",
        "Column",
        "Column Type",
        "Possible Task",
        "Score"

    ]]

    candidates = result.get(
        "target_candidates",
        []
    )

    for candidate in candidates:

        candidate_rows.append([

            candidate.get("rank", "N/A"),

            candidate.get("column", "N/A"),

            candidate.get("column_type", "N/A"),

            candidate.get("possible_task", "N/A"),

            candidate.get("score", "N/A")

        ])

    # Algorithm recommendations

    recommendation_rows = [[

        "Algorithm",

        "Reason",

        "Status"

    ]]

    recommendations = result.get(
        "recommendations",
        []
    )

    for recommendation in recommendations:

        recommendation_rows.append([

            recommendation.get(
                "algorithm",
                recommendation.get("name", "N/A")
            ),

            recommendation.get("reason", "N/A"),

            recommendation.get("status", "Recommended")

        ])

    return (
        summary_rows,
        candidate_rows,
        recommendation_rows,
        result
    )


# ==========================================================
# 8. MODEL PERFORMANCE EVALUATION
# ==========================================================

def create_model_performance_section(
    df,
    target_column,
    task_type,
    styles
):

    elements = []

    elements.append(
        Paragraph(
            "Model Performance Evaluation",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 10))

    if not target_column:

        elements.append(
            Paragraph(
                "Model evaluation was skipped because "
                "no target column was selected by the "
                "recommendation engine.",
                styles["Normal"]
            )
        )

        return elements

    try:

        # Run model evaluation

        model_result = evaluate_models(

            df=df,

            target_column=target_column,

            task_type=task_type

        )

        results = model_result["results"]

        best_algorithm = model_result["best_algorithm"]

        best_performance = model_result["best_performance"]

        detected_task = model_result["task_type"]

        elements.append(
            Paragraph(
                f"<b>Target Column:</b> "
                f"{escape(str(target_column))}",
                styles["Normal"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Task Type:</b> "
                f"{escape(str(detected_task).title())}",
                styles["Normal"]
            )
        )

        elements.append(Spacer(1, 10))

        # --------------------------------------------------
        # CLASSIFICATION RESULTS
        # --------------------------------------------------

        if detected_task == "classification":

            elements.append(
                Paragraph(
                    "Best Performing Algorithm",
                    styles["Heading3"]
                )
            )

            best_accuracy = best_performance["accuracy"]

            # Highlight best model

            best_rows = [

                ["Performance Metric", "Result"],

                ["Best Algorithm", best_algorithm],

                ["Best Accuracy", f"{best_accuracy}%"],

                [
                    "Precision",
                    f"{best_performance['precision']}%"
                ],

                [
                    "Recall",
                    f"{best_performance['recall']}%"
                ],

                [
                    "F1-Score",
                    f"{best_performance['f1_score']}%"
                ]

            ]

            elements.append(
                create_report_table(
                    best_rows,
                    styles,
                    col_widths=[3 * inch, 3 * inch]
                )
            )

            elements.append(Spacer(1, 15))

            elements.append(
                Paragraph(
                    "Model Performance Comparison",
                    styles["Heading3"]
                )
            )

            comparison_rows = [[

                "Algorithm",

                "Accuracy (%)",

                "Precision (%)",

                "Recall (%)",

                "F1-Score (%)",

                "Status"

            ]]

            for item in results:

                comparison_rows.append([

                    item.get("algorithm", "N/A"),

                    safe_value(item.get("accuracy")),

                    safe_value(item.get("precision")),

                    safe_value(item.get("recall")),

                    safe_value(item.get("f1_score")),

                    item.get("status", "N/A")

                ])

            elements.append(
                create_report_table(
                    comparison_rows,
                    styles,
                    font_size=7
                )
            )

        # --------------------------------------------------
        # REGRESSION RESULTS
        # --------------------------------------------------

        else:

            elements.append(
                Paragraph(
                    "Best Performing Algorithm",
                    styles["Heading3"]
                )
            )

            best_rows = [

                ["Performance Metric", "Result"],

                ["Best Algorithm", best_algorithm],

                [
                    "Best R² Score",
                    f"{best_performance['r2_score']}%"
                ],

                [
                    "Mean Absolute Error (MAE)",
                    best_performance["mae"]
                ],

                [
                    "Root Mean Squared Error (RMSE)",
                    best_performance["rmse"]
                ]

            ]

            elements.append(
                create_report_table(
                    best_rows,
                    styles,
                    col_widths=[3 * inch, 3 * inch]
                )
            )

            elements.append(Spacer(1, 15))

            elements.append(
                Paragraph(
                    "Regression Model Comparison",
                    styles["Heading3"]
                )
            )

            comparison_rows = [[

                "Algorithm",

                "R² Score (%)",

                "MAE",

                "RMSE",

                "Status"

            ]]

            for item in results:

                comparison_rows.append([

                    item.get("algorithm", "N/A"),

                    safe_value(item.get("r2_score")),

                    safe_value(item.get("mae")),

                    safe_value(item.get("rmse")),

                    item.get("status", "N/A")

                ])

            elements.append(
                create_report_table(
                    comparison_rows,
                    styles,
                    font_size=7
                )
            )

        elements.append(Spacer(1, 12))

        elements.append(
            Paragraph(
                f"<b>Training Rows:</b> "
                f"{model_result['training_rows']} "
                f"&nbsp;&nbsp; "
                f"<b>Testing Rows:</b> "
                f"{model_result['testing_rows']}",
                styles["Normal"]
            )
        )

        elements.append(Spacer(1, 8))

        elements.append(
            Paragraph(
                "Note: Model performance is calculated using "
                "a held-out test dataset. The best algorithm "
                "is selected according to the highest "
                "accuracy for classification or highest "
                "R² score for regression. Results depend "
                "on the selected target and dataset quality.",
                styles["SmallNote"]
            )
        )

    except Exception as error:

        elements.append(
            Paragraph(
                "<b>Model evaluation could not be completed:</b> "
                + escape(str(error)),
                styles["Normal"]
            )
        )

    return elements


# ==========================================================
# 9. MAIN PDF REPORT GENERATOR
# ==========================================================

def generate_pdf_report(

    df,

    eda_report=None,

    ai_insight=None,

    cleaning_report=None,

    output_path="reports/analysis_report.pdf"

):

    if df is None or not isinstance(df, pd.DataFrame):

        raise ValueError(
            "A valid pandas DataFrame is required."
        )

    if df.empty:

        raise ValueError(
            "Cannot generate a report from an empty dataset."
        )

    # Create output directories

    output_directory = os.path.dirname(output_path)

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    chart_folder = os.path.join(
        output_directory or ".",
        "report_charts"
    )

    os.makedirs(
        chart_folder,
        exist_ok=True
    )

    # Generate charts

    chart_paths = create_charts(
        df,
        chart_folder
    )

    # Create document

    document = SimpleDocTemplate(

        output_path,

        pagesize=A4,

        rightMargin=35,

        leftMargin=35,

        topMargin=40,

        bottomMargin=40

    )

    # ------------------------------------------------------
    # STYLES
    # ------------------------------------------------------

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(

            name="ReportTitle",

            parent=styles["Title"],

            fontSize=21,

            leading=26,

            textColor=colors.HexColor("#17365D"),

            alignment=1,

            spaceAfter=15

        )
    )

    styles.add(
        ParagraphStyle(

            name="TableHeader",

            parent=styles["Normal"],

            fontSize=7,

            leading=9,

            textColor=colors.white,

            fontName="Helvetica-Bold"

        )
    )

    styles.add(
        ParagraphStyle(

            name="TableCell",

            parent=styles["Normal"],

            fontSize=7,

            leading=9,

            wordWrap="CJK"

        )
    )

    styles.add(
        ParagraphStyle(

            name="SmallNote",

            parent=styles["Normal"],

            fontSize=8,

            leading=11,

            textColor=colors.HexColor("#555555")

        )
    )

    # ------------------------------------------------------
    # REPORT CONTENT
    # ------------------------------------------------------

    story = []

    # Title

    story.append(
        Paragraph(
            "AI DATA ANALYTICS REPORT",
            styles["ReportTitle"]
        )
    )

    story.append(
        Paragraph(
            "Autonomous Data Analytics and Insight Generation System",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 1: DATASET OVERVIEW
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "1. Dataset Overview",
            styles["Heading2"]
        )
    )

    overview_rows = [

        ["Parameter", "Value"],

        ["Dataset Name", "Uploaded Dataset"],

        ["Total Rows", len(df)],

        ["Total Columns", len(df.columns)],

        [
            "Numerical Columns",
            len(df.select_dtypes(include="number").columns)
        ],

        [
            "Categorical Columns",
            len(df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns)
        ]

    ]

    story.append(
        create_report_table(
            overview_rows,
            styles
        )
    )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 2: DATA QUALITY
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "2. Data Quality Findings",
            styles["Heading2"]
        )
    )

    quality_rows = create_data_quality(df)

    story.append(
        create_report_table(
            quality_rows,
            styles
        )
    )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 3: STATISTICAL METRICS
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "3. Important Statistical Metrics",
            styles["Heading2"]
        )
    )

    metric_rows = create_metric_table(df)

    story.append(
        create_report_table(
            metric_rows,
            styles
        )
    )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 4: IMPLEMENTED TECHNIQUES
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "4. Algorithms and Techniques Implemented",
            styles["Heading2"]
        )
    )

    techniques = get_implemented_techniques(
        df,
        cleaning_report
    )

    for technique in techniques:

        story.append(
            Paragraph(
                "• " + escape(str(technique)),
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))

    # ------------------------------------------------------
    # SECTION 5: ALGORITHM RECOMMENDATION ENGINE
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "5. Algorithm Recommendation Engine",
            styles["Heading2"]
        )
    )

    (
        summary_rows,
        candidate_rows,
        recommendation_rows,
        engine_result
    ) = create_algorithm_engine_section(df)

    story.append(
        Paragraph(
            "Dataset and Task Analysis",
            styles["Heading3"]
        )
    )

    story.append(
        create_report_table(
            summary_rows,
            styles
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Possible Target Columns",
            styles["Heading3"]
        )
    )

    story.append(
        create_report_table(
            candidate_rows,
            styles
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Recommended Algorithms",
            styles["Heading3"]
        )
    )

    story.append(
        create_report_table(
            recommendation_rows,
            styles
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Note: Algorithm recommendations are generated "
            "from dataset characteristics. Recommendation "
            "does not itself mean that a model has been trained.",
            styles["SmallNote"]
        )
    )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 6: MODEL PERFORMANCE EVALUATION
    # ------------------------------------------------------

    story.extend(
        create_model_performance_section(

            df=df,

            target_column=engine_result.get(
                "selected_target"
            ),

            task_type=engine_result.get(
                "detected_task"
            ),

            styles=styles

        )
    )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 7: AI-GENERATED INSIGHTS
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "7. AI-Generated Analysis and Insights",
            styles["Heading2"]
        )
    )

    if ai_insight:

        if isinstance(ai_insight, dict):

            for key, value in ai_insight.items():

                story.append(
                    Paragraph(
                        f"<b>{escape(str(key))}:</b> "
                        f"{escape(str(value))}",
                        styles["Normal"]
                    )
                )

                story.append(Spacer(1, 5))

        elif isinstance(ai_insight, list):

            for insight in ai_insight:

                story.append(
                    Paragraph(
                        "• " + escape(str(insight)),
                        styles["Normal"]
                    )
                )

        else:

            insight_text = str(ai_insight).replace(
                "\n",
                "<br/>"
            )

            story.append(
                Paragraph(
                    escape(insight_text),
                    styles["Normal"]
                )
            )

    else:

        story.append(
            Paragraph(
                "No AI-generated insights were provided "
                "for this report.",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 15))

    # ------------------------------------------------------
    # SECTION 8: DATA VISUALIZATIONS
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "8. Data Visualizations",
            styles["Heading2"]
        )
    )

    if chart_paths:

        for chart_path in chart_paths:

            if os.path.exists(chart_path):

                story.append(
                    Image(
                        chart_path,
                        width=6 * inch,
                        height=3.5 * inch
                    )
                )

                story.append(Spacer(1, 12))

    else:

        story.append(
            Paragraph(
                "No suitable visualizations were generated.",
                styles["Normal"]
            )
        )

    # ------------------------------------------------------
    # SECTION 9: CONCLUSION
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "9. Conclusion",
            styles["Heading2"]
        )
    )

    conclusion = (
        "The AI Data Analytics System performed automated "
        "dataset profiling, data quality analysis, statistical "
        "analysis, algorithm recommendation and data "
        "visualization. Where a suitable target column was "
        "identified, machine learning algorithms were trained "
        "and evaluated using test data. The model performance "
        "results provide a comparison of the evaluated "
        "algorithms and identify the best-performing model "
        "according to the selected evaluation metric."
    )

    story.append(
        Paragraph(
            conclusion,
            styles["Normal"]
        )
    )

    # ------------------------------------------------------
    # BUILD PDF
    # ------------------------------------------------------

    document.build(story)

    return output_path


# ==========================================================
# 10. TEST
# ==========================================================

if __name__ == "__main__":

    print("Report Generator Loaded Successfully.")
