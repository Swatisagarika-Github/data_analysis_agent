
import streamlit as st
import pandas as pd
import os

from module.data_loader import load_dataset
from module.data_understanding import understand_dataset
from module.data_cleaning import clean_dataset
from agents.eda_agent import run_eda_agent
from agents.visualization_agent import run_visualization_agent
from module.chart_generator import generate_chart
from report_generator_with_engine import generate_pdf_report


# --------------------------------
# Helper: Safe DataFrame Display
# --------------------------------

def make_display_safe(df):

    display_df = df.copy()

    for column in display_df.columns:
        display_df[column] = display_df[column].map(
            lambda value: (
                "" if pd.isna(value) else str(value)
            )
        )

    return display_df


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)


# --------------------------------
# Session State
# --------------------------------

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None

if "report_bytes" not in st.session_state:
    st.session_state.report_bytes = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = None


# --------------------------------
# Professional SaaS Dashboard Theme
# --------------------------------

st.markdown("""
<style>

/* Main canvas */

.stApp {
    background: #F3F6FC;
    color: #172B4D;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Typography */

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3 {
    color: #14243A !important;
    letter-spacing: -0.35px;
}

p, label, .stMarkdown {
    color: #40516B;
}

/* Navy sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101D35 0%, #182E50 100%);
}

[data-testid="stSidebar"] * {
    color: #EAF1FF !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B9C8DF !important;
}

/* Gradient page header */

.dashboard-hero {
    background: linear-gradient(115deg, #10243F 0%, #1D4E89 55%, #3867D6 100%);
    padding: 30px 34px;
    border-radius: 20px;
    margin: 0 0 26px 0;
    box-shadow: 0 12px 28px rgba(27, 61, 112, 0.16);
}

.dashboard-hero h1 {
    color: #FFFFFF !important;
    font-size: 32px !important;
    font-weight: 750 !important;
    margin: 0 0 8px 0;
}

.dashboard-hero p {
    color: #DCE9FF !important;
    font-size: 15px;
    margin: 0;
}

.hero-tag {
    display: inline-block;
    color: #EAF2FF;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4EAF3;
    border-radius: 16px;
    padding: 17px 19px;
    box-shadow: 0 5px 16px rgba(25, 48, 82, 0.055);
    min-height: 112px;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #183B66 !important;
    font-size: 27px !important;
    font-weight: 750 !important;
}

/* Section and content surfaces */

[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stFileUploader"],
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E4EAF3;
    border-radius: 15px;
}

[data-testid="stDataFrame"] {
    padding: 7px;
}

[data-testid="stFileUploader"] {
    padding: 14px;
}

[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E4EAF3;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 5px 16px rgba(25, 48, 82, 0.045);
}

/* Buttons */

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #2563EB, #4F46E5);
    color: #FFFFFF;
    border: 0;
    border-radius: 10px;
    padding: 0.58rem 1.1rem;
    font-weight: 650;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.16);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    color: #FFFFFF;
    border: 0;
    background: linear-gradient(100deg, #1D4ED8, #4338CA);
}

/* Inputs and separators */

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    border-radius: 9px;
}

hr {
    border-color: #E2E8F0;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.section-kicker {
    color: #3267C8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-bottom: 3px;
}

.small-muted {
    color: #718096;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# Section Heading Helper
# --------------------------------

def section_heading(title, subtitle=None):

    st.markdown(
        f'<div class="section-kicker">WORKSPACE</div>'
        f'<h2 style="margin-top:0;">{title}</h2>',
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f'<p class="small-muted" style="margin-top:-8px;">'
            f'{subtitle}</p>',
            unsafe_allow_html=True
        )


# --------------------------------
# Application Sidebar
# --------------------------------

with st.sidebar:

    st.markdown("## 📊 AI ANALYST")

    st.caption("INTELLIGENT ANALYTICS PLATFORM")

    st.divider()

    st.markdown("### Project Workspace")

    st.markdown(
        """
        **01** · Dataset Upload

        **02** · Data Understanding

        **03** · Data Cleaning

        **04** · PDF Report
        """
    )

    st.divider()

    st.info(
        "Upload a CSV or Excel file to begin exploring your data."
    )

    st.markdown(
        '<p class="small-muted">'
        'Final-year project · Python + Streamlit'
        '</p>',
        unsafe_allow_html=True
    )


# --------------------------------
# Application Header
# --------------------------------

st.markdown(
    """
    <div class="dashboard-hero">
        <div class="hero-tag">INTELLIGENT ANALYTICS WORKSPACE</div>
        <h1>AI-Powered Data Analyst</h1>
        <p>Transform raw datasets into clean, understandable and analysis-ready data.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------
# Dataset Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Choose your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    try:

        # =================================
        # NEW DATASET DETECTION
        # =================================

        current_upload_key = (
            uploaded_file.name,
            uploaded_file.size
        )

        if st.session_state.upload_key != current_upload_key:

            st.session_state.upload_key = current_upload_key

            st.session_state.cleaned_df = None

            st.session_state.cleaning_report = None

            st.session_state.report_bytes = None


        # =================================
        # PHASE 1: DATASET UPLOAD
        # =================================

        df = load_dataset(uploaded_file)

        if df is None or df.empty:

            st.error("Uploaded dataset is empty.")

            st.stop()

        st.success(
            f"Dataset uploaded successfully · {uploaded_file.name}"
        )

        section_heading(
            "01 · Dataset Preview",
            "A quick look at the first records in your uploaded file."
        )

        st.dataframe(
            make_display_safe(df.head(10)),
            width="stretch"
        )

        col1, col2 = st.columns(2)

        col1.metric("Total Rows", len(df))

        col2.metric("Total Columns", len(df.columns))


        # =================================
        # PHASE 2: DATA UNDERSTANDING
        # =================================

        st.divider()

        section_heading(
            "02 · Data Understanding",
            "Review the structure and quality of your dataset."
        )

        profile = understand_dataset(df)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Numerical Columns",
            len(profile["numerical_columns"])
        )

        col2.metric(
            "Categorical Columns",
            len(profile["categorical_columns"])
        )

        col3.metric(
            "Duplicate Rows",
            profile["duplicate_rows"]
        )

        # Missing Values

        st.subheader("Missing Values")

        missing_df = pd.DataFrame(
            list(profile["missing_values"].items()),
            columns=["Column Name", "Missing Values"]
        )

        st.dataframe(
            make_display_safe(missing_df),
            width="stretch"
        )

        # Unique Values

        st.subheader("Unique Values")

        unique_df = pd.DataFrame(
            list(profile["unique_values"].items()),
            columns=["Column Name", "Unique Values"]
        )

        st.dataframe(
            make_display_safe(unique_df),
            width="stretch"
        )

        # Data Types

        st.subheader("Data Types")

        dtype_df = pd.DataFrame(
            list(profile["data_types"].items()),
            columns=["Column Name", "Data Type"]
        )

        st.dataframe(
            make_display_safe(dtype_df),
            width="stretch"
        )

        # Column Groups

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Numerical Columns")

            st.write(profile["numerical_columns"])

        with col2:

            st.subheader("Categorical Columns")

            st.write(profile["categorical_columns"])


        # =================================
        # PHASE 3: DATA CLEANING AGENT
        # =================================

        st.divider()

        section_heading(
            "03 · Data Cleaning Studio",
            "Choose the cleaning operations you want to apply."
        )

        st.write(
            "Configure the cleaning operations below. "
            "Your original uploaded dataset will remain unchanged."
        )

        numerical_columns = profile["numerical_columns"]

        categorical_columns = profile["categorical_columns"]

        # Date candidates

        date_candidates = [
            column
            for column in df.columns
            if (
                pd.api.types.is_object_dtype(df[column])
                or isinstance(df[column].dtype, pd.StringDtype)
            )
        ]

        # Cleaning controls

        with st.form("data_cleaning_form"):

            st.subheader("1. Missing Value Handling")

            numeric_strategy = st.selectbox(
                "Numerical missing values",
                [
                    "Median",
                    "Mean",
                    "Keep missing values",
                    "Drop rows with missing values"
                ]
            )

            categorical_strategy = st.selectbox(
                "Categorical missing values",
                [
                    "Mode",
                    "Fill with Unknown",
                    "Keep missing values",
                    "Drop rows with missing values"
                ]
            )

            st.divider()

            st.subheader("2. Duplicate Records")

            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                value=True
            )

            st.divider()

            st.subheader("3. Outlier Detection")

            outlier_columns = st.multiselect(
                "Select numerical columns for IQR outlier detection",
                options=numerical_columns,
                default=numerical_columns
            )

            outlier_action = st.radio(
                "What should happen to detected outliers?",
                ["Keep", "Remove rows"],
                horizontal=True
            )

            st.divider()

            st.subheader("4. Category Standardization")

            standardize_categories = st.checkbox(
                "Standardize categorical values",
                value=False
            )

            category_columns = st.multiselect(
                "Select categorical columns",
                options=categorical_columns,
                default=[]
            )

            category_format = st.selectbox(
                "Category formatting",
                [
                    "Trim spaces only",
                    "Title Case",
                    "Lowercase"
                ]
            )

            st.divider()

            st.subheader("5. Date Conversion")

            date_columns = st.multiselect(
                "Select columns to convert into dates",
                options=date_candidates,
                default=[]
            )

            st.divider()

            clean_button = st.form_submit_button(
                "Run Data Cleaning",
                width="stretch"
            )


        # =================================
        # RUN CLEANING
        # =================================

        if clean_button:

            cleaned_df, report = clean_dataset(
                df=df,
                numeric_strategy=numeric_strategy,
                categorical_strategy=categorical_strategy,
                remove_duplicates=remove_duplicates,
                outlier_action=outlier_action,
                outlier_columns=outlier_columns,
                standardize_categories=standardize_categories,
                category_columns=category_columns,
                category_format=category_format,
                date_columns=date_columns
            )

            # Save results across Streamlit reruns

            st.session_state.cleaned_df = cleaned_df

            st.session_state.cleaning_report = report

            st.session_state.report_bytes = None

            st.success("Data cleaning completed!")


        # =================================
        # DISPLAY SAVED CLEANING RESULTS
        # =================================

        if st.session_state.cleaned_df is not None:

            cleaned_df = st.session_state.cleaned_df

            report = st.session_state.cleaning_report

            # Before and After Comparison

            st.subheader("Cleaning Summary")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Before Cleaning**")

                st.metric(
                    "Rows",
                    report["rows_before"]
                )

                st.metric(
                    "Missing Values",
                    report["missing_before"]
                )

                st.metric(
                    "Duplicate Rows",
                    report["duplicates_before"]
                )

            with col2:

                st.markdown("**After Cleaning**")

                st.metric(
                    "Rows",
                    report["rows_after"]
                )

                st.metric(
                    "Missing Values",
                    report["missing_after"]
                )

                st.metric(
                    "Duplicate Rows",
                    report["duplicates_after"]
                )

            # Outlier Report

            st.subheader("Outlier Detection Report")

            if report["outliers"]:

                outlier_df = pd.DataFrame([

                    {
                        "Column": column,

                        "Outlier Count": details["count"],

                        "Lower Limit": round(
                            details["lower_limit"], 2
                        ),

                        "Upper Limit": round(
                            details["upper_limit"], 2
                        )
                    }

                    for column, details in report["outliers"].items()

                ])

                st.dataframe(
                    make_display_safe(outlier_df),
                    width="stretch"
                )

            else:

                st.info(
                    "No numerical columns were selected for outlier detection."
                )

            # Cleaning Actions

            st.subheader("Cleaning Actions")

            if report["cleaning_actions"]:

                for action in report["cleaning_actions"]:

                    st.write("✓", action)

            else:

                st.info(
                    "No cleaning operations were required."
                )

            # Date Conversion Report

            if report["converted_dates"]:

                st.subheader("Date Conversion Report")

                date_report_df = pd.DataFrame(
                    list(report["converted_dates"].items()),
                    columns=["Column", "New Invalid Dates"]
                )

                st.dataframe(
                    make_display_safe(date_report_df),
                    width="stretch"
                )

            # Cleaned Dataset

            st.divider()

            st.subheader("Cleaned Dataset Preview")

            st.dataframe(
                make_display_safe(cleaned_df.head(20)),
                width="stretch"
            )

            # Download Cleaned Dataset

            csv_data = cleaned_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Download Cleaned Dataset",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                width="stretch"
            )


            # =================================
            # PHASE 4: PDF REPORT GENERATION
            # =================================

            st.divider()

            section_heading(
                "04 · Download Analysis Report",
                "Generate a professional PDF report containing "
                "dataset analysis, data quality findings, "
                "algorithm recommendations and visualizations."
            )

            st.markdown(
                """
                <div style="
                    background: white;
                    border: 1px solid #E4EAF3;
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 15px;
                ">
                    <h3 style="margin-top:0;">
                        📄 AI Data Analytics Report
                    </h3>

                    <p>
                        Your cleaned dataset is ready. Generate a PDF
                        report to save and share your analysis results.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "📄 Generate PDF Report",
                type="primary",
                width="stretch"
            ):

                try:

                    with st.spinner(
                        "Generating your PDF report..."
                    ):

                        # Generate PDF

                        report_path = generate_pdf_report(
                            df=cleaned_df,
                            eda_report={},
                            ai_insight=None,
                            cleaning_report=report,
                            output_path="reports/analysis_report.pdf"
                        )

                        # Read PDF

                        with open(report_path, "rb") as pdf_file:

                            st.session_state.report_bytes = (
                                pdf_file.read()
                            )

                    st.success(
                        "PDF report generated successfully!"
                    )

                except Exception as report_error:

                    st.error(
                        f"PDF generation failed: {report_error}"
                    )

                    st.exception(report_error)


            # =================================
            # PDF DOWNLOAD BUTTON
            # =================================

            if st.session_state.report_bytes is not None:

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state.report_bytes,
                    file_name="AI_Data_Analytics_Report.pdf",
                    mime="application/pdf",
                    width="stretch"
                )

                st.success(
                    "Your PDF report is ready to download!"
                )


    except Exception as e:

        st.error(f"An error occurred: {e}")

        st.exception(e)
