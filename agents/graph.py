
from langgraph.graph import StateGraph, START, END

from agents.state import AnalyticsState

from module.data_understanding import understand_dataset
from module.data_cleaning import clean_dataset

from agents.eda_agent import run_eda_agent
from agents.visualization_agent import run_visualization_agent
from agents.insight_agent import run_insight_agent
from agents.report_agent import run_report_agent


# -----------------------------------------
# 1. DATA UNDERSTANDING NODE
# -----------------------------------------

def understanding_node(state: AnalyticsState):

    df = state["raw_df"]

    understanding_report = understand_dataset(df)

    return {
        "understanding_report": understanding_report,
        "current_step": "understanding_completed"
    }


# -----------------------------------------
# 2. DATA CLEANING NODE
# -----------------------------------------

def cleaning_node(state: AnalyticsState):

    df = state["raw_df"]

    cleaned_df, cleaning_report = clean_dataset(df)

    return {
        "cleaned_df": cleaned_df,
        "cleaning_report": cleaning_report,
        "current_step": "cleaning_completed"
    }


# -----------------------------------------
# 3. SELECT DATA FOR ANALYSIS
# -----------------------------------------

def get_analysis_df(state: AnalyticsState):

    if state.get("cleaned_df") is not None:
        return state["cleaned_df"]

    return state["raw_df"]


# -----------------------------------------
# 4. EDA NODE
# -----------------------------------------

def eda_node(state: AnalyticsState):

    df = get_analysis_df(state)

    eda_report = run_eda_agent(df)

    return {
        "eda_report": eda_report,
        "current_step": "eda_completed"
    }


# -----------------------------------------
# 5. VISUALIZATION NODE
# -----------------------------------------

def visualization_node(state: AnalyticsState):

    df = get_analysis_df(state)

    visualization_report = run_visualization_agent(df)

    return {
        "visualization_report": visualization_report,
        "current_step": "visualization_completed"
    }


# -----------------------------------------
# 6. AI INSIGHT NODE
# -----------------------------------------

def insight_node(state: AnalyticsState):

    eda_report = state["eda_report"]

    ai_insight = run_insight_agent(eda_report)

    return {
        "ai_insight": ai_insight,
        "current_step": "insights_completed"
    }


# -----------------------------------------
# 7. PDF REPORT NODE
# -----------------------------------------

def report_node(state: AnalyticsState):

    df = get_analysis_df(state)

    report_path = run_report_agent(
        df=df,
        eda_report=state["eda_report"],
        ai_insight=state.get("ai_insight"),
        cleaning_report = state.get("cleaning_report")
    )

    return {
        "report_path": report_path,
        "current_step": "report_completed"
    }


# -----------------------------------------
# 8. CLEANING APPROVAL ROUTER
# -----------------------------------------

def cleaning_router(state: AnalyticsState):

    if state.get("cleaning_approved", False):
        return "cleaning"

    return "eda"


# -----------------------------------------
# 9. BUILD LANGGRAPH
# -----------------------------------------

def build_analytics_graph():

    workflow = StateGraph(AnalyticsState)

    workflow.add_node("understanding", understanding_node)
    workflow.add_node("cleaning", cleaning_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("visualization", visualization_node)
    workflow.add_node("insights", insight_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "understanding")

    workflow.add_conditional_edges(
        "understanding",
        cleaning_router,
        {
            "cleaning": "cleaning",
            "eda": "eda"
        }
    )

    workflow.add_edge("cleaning", "eda")
    workflow.add_edge("eda", "visualization")
    workflow.add_edge("visualization", "insights")
    workflow.add_edge("insights", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


analytics_graph = build_analytics_graph()
