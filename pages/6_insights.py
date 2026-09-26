import os
import traceback

import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insights & Root Cause Analysis",
    page_icon="💡",
    layout="wide",
)


# ============================================================
# 2. LOAD CSS
# ============================================================

if os.path.exists("assets/style.css"):
    try:
        with open("assets/style.css", encoding="utf-8") as f:
            css = f.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )

    except Exception:
        pass


# ============================================================
# 3. HEADER
# ============================================================

st.title("💡 Insights & Root Cause Analysis")

st.write(
    "Explore frequent alarms, site behavior, and severity patterns."
)


# ============================================================
# 4. GET DATA
# ============================================================

df = None


# First try preprocessed data
if (
    "preprocessed_data" in st.session_state
    and st.session_state.preprocessed_data is not None
):

    df = st.session_state.preprocessed_data.copy()


# Otherwise try raw uploaded data
elif (
    "raw_data" in st.session_state
    and st.session_state.raw_data is not None
):

    df = st.session_state.raw_data.copy()


# If no data is available
if df is None:

    st.warning(
        "⚠️ No data is available for analysis."
    )

    st.info(
        "Please go to Data Upload and upload your network alarm CSV first."
    )

    st.stop()


# ============================================================
# 5. REMOVE DUPLICATE COLUMNS
# ============================================================

df = df.loc[
    :,
    ~df.columns.duplicated(keep="first")
].copy()


# ============================================================
# 6. NORMALIZE COLUMN NAMES
# ============================================================

# Keep original columns but create a mapping
column_map = {}

for column in df.columns:

    clean_name = str(column).strip().lower()

    column_map[clean_name] = column


# ============================================================
# 7. FIND IMPORTANT COLUMNS
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        key = name.strip().lower()

        if key in column_map:

            return column_map[key]

    return None


alarm_col = find_column(
    [
        "Alarm Type",
        "alarm_type",
        "alarm type",
        "Alarm Name",
        "alarm_name",
    ]
)


site_col = find_column(
    [
        "Site Name",
        "site_id",
        "site",
        "Site",
        "NODE",
        "node",
        "Alarmed Object Name",
    ]
)


severity_col = find_column(
    [
        "Severity",
        "severity",
        "Previous Severity",
        "previous_severity",
    ]
)


cause_col = find_column(
    [
        "Probable Cause Merged",
        "Probable Cause",
        "probable_cause",
        "cause",
        "Specific Problem",
        "specific_problem",
    ]
)


duration_col = find_column(
    [
        "Life Span (minutes)",
        "duration_minutes",
        "Duration",
        "duration",
    ]
)


timestamp_col = find_column(
    [
        "First Time Detected Clean",
        "First Time Detected",
        "timestamp",
        "Timestamp",
    ]
)


# ============================================================
# 8. SHOW DATA INFORMATION
# ============================================================

with st.expander("🔍 View Detected Columns"):

    st.write(
        "Columns found in the current dataset:"
    )

    st.write(
        list(df.columns)
    )

    st.write(
        {
            "Alarm Column": alarm_col,
            "Site Column": site_col,
            "Severity Column": severity_col,
            "Cause Column": cause_col,
            "Duration Column": duration_col,
            "Timestamp Column": timestamp_col,
        }
    )


# ============================================================
# 9. ANALYSIS FOCUS
# ============================================================

st.markdown("---")

st.subheader("🔎 Select Analysis Focus")

analysis_focus = st.radio(
    "Choose an analysis:",
    [
        "Alarm Pattern Analysis",
        "Site-Specific Analysis",
        "Severity Root Cause",
    ],
    horizontal=True,
)


# ============================================================
# 10. ALARM PATTERN ANALYSIS
# ============================================================

if analysis_focus == "Alarm Pattern Analysis":

    st.subheader("📊 Alarm Pattern Analysis")


    if alarm_col is None:

        st.error(
            "❌ Alarm Type column was not found."
        )

        st.info(
            "Available columns are shown above."
        )

        st.stop()


    alarm_counts = (
        df[alarm_col]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )


    alarm_counts.columns = [
        "Alarm Type",
        "Count",
    ]


    if alarm_counts.empty:

        st.warning(
            "No alarm data is available."
        )

    else:

        st.markdown(
            "### 🔔 Most Frequent Alarms"
        )


        st.dataframe(
            alarm_counts.head(20),
            use_container_width=True,
        )


        st.bar_chart(
            alarm_counts.head(10).set_index(
                "Alarm Type"
            )
        )


        st.markdown(
            "### 📌 Key Observation"
        )


        top_alarm = alarm_counts.iloc[0]

        st.info(
            f"The most frequent alarm is "
            f"**{top_alarm['Alarm Type']}**, "
            f"with **{top_alarm['Count']} occurrences**."
        )


# ============================================================
# 11. SITE-SPECIFIC ANALYSIS
# ============================================================

elif analysis_focus == "Site-Specific Analysis":

    st.subheader("🏢 Site-Specific Analysis")


    if site_col is None:

        st.error(
            "❌ Site column was not found."
        )

        st.stop()


    site_counts = (
        df[site_col]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )


    site_counts.columns = [
        "Site",
        "Alarm Count",
    ]


    if site_counts.empty:

        st.warning(
            "No site information is available."
        )

    else:

        st.markdown(
            "### 🏢 Alarm Distribution by Site"
        )


        st.dataframe(
            site_counts.head(20),
            use_container_width=True,
        )


        st.bar_chart(
            site_counts.head(10).set_index(
                "Site"
            )
        )


        st.markdown(
            "### 🔎 Select a Site"
        )


        site_list = site_counts["Site"].tolist()


        selected_site = st.selectbox(
            "Choose Site:",
            site_list,
        )


        selected_data = df[
            df[site_col].astype(str)
            == str(selected_site)
        ].copy()


        st.metric(
            "Total Alarms",
            len(selected_data),
        )


        if alarm_col is not None:

            site_alarm_counts = (
                selected_data[alarm_col]
                .dropna()
                .astype(str)
                .value_counts()
                .reset_index()
            )


            site_alarm_counts.columns = [
                "Alarm Type",
                "Count",
            ]


            st.markdown(
                "### 🔔 Alarms at Selected Site"
            )


            st.dataframe(
                site_alarm_counts,
                use_container_width=True,
            )


# ============================================================
# 12. SEVERITY ROOT CAUSE
# ============================================================

elif analysis_focus == "Severity Root Cause":

    st.subheader("🚨 Severity Root Cause Analysis")


    if severity_col is None:

        st.error(
            "❌ Severity column was not found."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(df.columns)
        )

        st.info(
            "The page searched for both 'Severity' and 'severity'."
        )

        st.stop()


    # --------------------------------------------------------
    # Severity Distribution
    # --------------------------------------------------------

    levels = (
        df[severity_col]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )


    levels.columns = [
        "Severity",
        "Count",
    ]


    st.markdown(
        "### 🚨 Severity Distribution"
    )


    if levels.empty:

        st.warning(
            "No severity values are available."
        )

    else:

        st.dataframe(
            levels,
            use_container_width=True,
        )


        st.bar_chart(
            levels.set_index("Severity")
        )


    # --------------------------------------------------------
    # Severity + Alarm Type
    # --------------------------------------------------------

    if alarm_col is not None:

        st.markdown(
            "### 🔗 Severity by Alarm Type"
        )


        severity_alarm = pd.crosstab(
            df[alarm_col].astype(str),
            df[severity_col].astype(str),
        )


        st.dataframe(
            severity_alarm,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Severity + Site
    # --------------------------------------------------------

    if site_col is not None:

        st.markdown(
            "### 🏢 Severity by Site"
        )


        severity_site = pd.crosstab(
            df[site_col].astype(str),
            df[severity_col].astype(str),
        )


        st.dataframe(
            severity_site,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Severity + Cause
    # --------------------------------------------------------

    if cause_col is not None:

        st.markdown(
            "### 🧠 Probable Causes by Severity"
        )


        cause_severity = pd.crosstab(
            df[cause_col].astype(str),
            df[severity_col].astype(str),
        )


        st.dataframe(
            cause_severity,
            use_container_width=True,
        )


# ============================================================
# 13. DATA SUMMARY
# ============================================================

st.markdown("---")

st.subheader("📋 Dataset Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)


with summary_col1:

    st.metric(
        "Total Records",
        len(df),
    )


with summary_col2:

    if site_col is not None:

        st.metric(
            "Unique Sites",
            df[site_col].nunique(),
        )

    else:

        st.metric(
            "Unique Sites",
            0,
        )


with summary_col3:

    if alarm_col is not None:

        st.metric(
            "Alarm Types",
            df[alarm_col].nunique(),
        )

    else:

        st.metric(
            "Alarm Types",
            0,
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Network Alarm Prediction System — Insights & Root Cause Analysis"
)