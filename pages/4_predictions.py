import os
import traceback

import pandas as pd
import streamlit as st

from models.model_training import (
    chained_predict,
    batch_chained_predict,
    load_all_models,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Alarm Predictions",
    page_icon="🔮",
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
# 3. PAGE HEADER
# ============================================================

st.title("🔮 Alarm Predictions")

st.write(
    "Predict alarm type, probable cause, duration, and severity "
    "using the trained machine learning models."
)


# ============================================================
# 4. LOAD TRAINED MODELS
# ============================================================

try:

    if (
        "model_bundle" in st.session_state
        and st.session_state.model_bundle is not None
    ):

        model_dict, encoders = st.session_state.model_bundle

    else:

        model_dict, encoders = load_all_models("models")

        st.session_state.model_bundle = (
            model_dict,
            encoders,
        )

except Exception:

    st.error("❌ Could not load the trained models.")

    st.code(
        traceback.format_exc(),
        language="text",
    )

    st.warning(
        "Please go to ML Modeling and train all models first."
    )

    st.stop()


# ============================================================
# 5. CHECK MODEL INFORMATION
# ============================================================

if not isinstance(model_dict, dict):

    st.error("❌ Invalid model information.")

    st.info(
        "Please retrain the models from the ML Modeling page."
    )

    st.stop()


if "input_features" not in model_dict:

    st.error(
        "❌ Model input features are missing."
    )

    st.info(
        "Please retrain all models from the ML Modeling page."
    )

    st.stop()


features = model_dict["input_features"]


# ============================================================
# 6. CHECK PREPROCESSED DATA
# ============================================================

if (
    "preprocessed_data" not in st.session_state
    or st.session_state.preprocessed_data is None
):

    st.warning(
        "⚠️ No preprocessed data is available."
    )

    st.info(
        "Go to ML Modeling → Train All Models, "
        "then return to this Predictions page."
    )

    st.stop()


df = st.session_state.preprocessed_data.copy()


# ============================================================
# 7. REMOVE DUPLICATE COLUMNS
# ============================================================

df = df.loc[
    :,
    ~df.columns.duplicated(keep="first")
].copy()


# ============================================================
# 8. BASIC DATA INFORMATION
# ============================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Records",
        len(df),
    )


with col2:

    if "Site Name" in df.columns:

        site_count = df["Site Name"].nunique()

    else:

        site_count = 0

    st.metric(
        "Unique Sites",
        site_count,
    )


with col3:

    model_names = [
        "alarm_type_clf",
        "cause_clf",
        "duration_reg",
        "severity_clf",
        "timestamp_reg",
    ]

    trained_count = 0

    for model_name in model_names:

        if model_dict.get(model_name) is not None:

            trained_count += 1

    st.metric(
        "Trained Models",
        trained_count,
    )


# ============================================================
# 9. PREDICTION TYPE
# ============================================================

st.markdown("---")

st.subheader("⚙️ Prediction Configuration")

prediction_type = st.radio(
    "Select prediction type:",
    [
        "Top 10 Predictions",
        "Prediction for One Site",
        "Predictions for All Sites",
    ],
    horizontal=True,
)


# ============================================================
# 10. TOP 10 PREDICTIONS
# ============================================================

if prediction_type == "Top 10 Predictions":

    st.subheader("🔮 Top 10 Chained Predictions")

    try:

        alarm_features = features.get(
            "alarm_type",
            [],
        )

        if not alarm_features:

            st.error(
                "❌ Alarm Type model features were not found."
            )

            st.stop()


        missing_columns = []

        for column in alarm_features:

            if column not in df.columns:

                missing_columns.append(column)


        if missing_columns:

            st.error(
                "❌ Some required model columns are missing."
            )

            st.write(
                "Missing columns:"
            )

            st.write(
                missing_columns
            )

            st.stop()


        X = df[
            alarm_features
        ].tail(10).copy()


        results_df = batch_chained_predict(
            X,
            model_dict,
            encoders,
        )


        if results_df is None:

            st.warning(
                "No predictions were generated."
            )

        elif results_df.empty:

            st.warning(
                "No predictions were generated."
            )

        else:

            st.dataframe(
                results_df,
                use_container_width=True,
            )


            csv_data = results_df.to_csv(
                index=False
            )


            st.download_button(
                label="📥 Download Top 10 Predictions",
                data=csv_data,
                file_name="top_10_predictions.csv",
                mime="text/csv",
            )


    except Exception:

        st.error(
            "❌ Error while generating Top 10 predictions."
        )

        st.code(
            traceback.format_exc(),
            language="text",
        )


# ============================================================
# 11. PREDICTION FOR ONE SITE
# ============================================================

elif prediction_type == "Prediction for One Site":

    st.subheader("🏢 Prediction for One Site")


    if "Site Name" not in df.columns:

        st.error(
            "❌ 'Site Name' column is missing."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(df.columns)
        )

        st.stop()


    site_encoder = encoders.get(
        "Site Name"
    )


    if site_encoder is None:

        st.error(
            "❌ Site Name encoder was not found."
        )

        st.info(
            "Please retrain the models."
        )

        st.stop()


    site_names = list(
        site_encoder.classes_
    )


    if len(site_names) == 0:

        st.warning(
            "No sites are available."
        )

        st.stop()


    selected_site = st.selectbox(
        "Select Site:",
        site_names,
    )


    try:

        encoded_site = site_encoder.transform(
            [selected_site]
        )[0]

    except Exception:

        st.error(
            "❌ Could not encode the selected site."
        )

        st.stop()


    site_data = df[
        df["Site Name"] == encoded_site
    ].copy()


    if site_data.empty:

        st.warning(
            "No records were found for this site."
        )

        st.stop()


    latest_row = site_data.tail(1)


    try:

        result = chained_predict(
            latest_row,
            model_dict,
            encoders,
        )

    except Exception:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            traceback.format_exc(),
            language="text",
        )

        st.stop()


    st.success(
        f"Prediction generated for {selected_site}"
    )


    st.markdown(
        "### 🔮 Prediction Results"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Predicted Alarm Type",
            result.get(
                "Predicted Alarm Type",
                "N/A",
            ),
        )


        st.metric(
            "Predicted Probable Cause",
            result.get(
                "Predicted Probable Cause",
                "N/A",
            ),
        )


    with col2:

        st.metric(
            "Predicted Severity",
            result.get(
                "Predicted Severity",
                "N/A",
            ),
        )


        duration = result.get(
            "Predicted Duration (minutes)",
            "N/A",
        )


        try:

            duration_text = (
                f"{float(duration):.2f} minutes"
            )

        except Exception:

            duration_text = str(duration)


        st.metric(
            "Predicted Duration",
            duration_text,
        )


    st.markdown(
        "### 📋 Detailed Prediction"
    )


    result_df = pd.DataFrame(
        [result]
    )


    st.dataframe(
        result_df,
        use_container_width=True,
    )


    with st.expander(
        "View Latest Record Used"
    ):

        st.dataframe(
            latest_row,
            use_container_width=True,
        )


# ============================================================
# 12. PREDICTIONS FOR ALL SITES
# ============================================================

elif prediction_type == "Predictions for All Sites":

    st.subheader(
        "🌐 Predictions for All Sites"
    )


    if "Site Name" not in df.columns:

        st.error(
            "❌ 'Site Name' column is missing."
        )

        st.stop()


    unique_sites = (
        df["Site Name"]
        .dropna()
        .unique()
    )


    if len(unique_sites) == 0:

        st.warning(
            "No sites were found."
        )

        st.stop()


    st.write(
        f"Generating predictions for "
        f"{len(unique_sites)} sites."
    )


    prediction_rows = []


    progress_bar = st.progress(0)


    total_sites = len(unique_sites)


    for number, site_value in enumerate(
        unique_sites,
        start=1,
    ):

        try:

            site_data = df[
                df["Site Name"] == site_value
            ].copy()


            if site_data.empty:

                continue


            latest_row = site_data.tail(1)


            result = chained_predict(
                latest_row,
                model_dict,
                encoders,
            )


            result["Site"] = site_value


            prediction_rows.append(
                result
            )


        except Exception as error:

            prediction_rows.append(
                {
                    "Site": site_value,
                    "Prediction Error": str(error),
                }
            )


        progress_bar.progress(
            number / total_sites
        )


    progress_bar.empty()


    if len(prediction_rows) == 0:

        st.warning(
            "No predictions could be generated."
        )

    else:

        results_df = pd.DataFrame(
            prediction_rows
        )


        st.markdown(
            "### 📊 Site Predictions"
        )


        st.dataframe(
            results_df,
            use_container_width=True,
        )


        csv_data = results_df.to_csv(
            index=False
        )


        st.download_button(
            label="📥 Download All Predictions",
            data=csv_data,
            file_name="all_site_predictions.csv",
            mime="text/csv",
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Network Alarm Prediction System"
)