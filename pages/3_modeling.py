import os
import traceback

import pandas as pd
import streamlit as st

from models.model_training import (
    preprocess_alarm_data,
    train_all_models,
    save_all_models
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Modeling",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

if os.path.exists("assets/style.css"):

    with open(
        "assets/style.css",
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# CHECK DATA
# ============================================================

if (
    "raw_data" not in st.session_state
    or st.session_state.raw_data is None
):

    st.warning(
        "No raw data available. "
        "Please upload or load sample data "
        "from the Data Upload page."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header">
        <div class="logo">🤖</div>

        <div class="title-container">

            <h1 class="main-title">
                ML Model Training
            </h1>

            <p class="subtitle">
                Train all models for multi-step
                network alarm prediction
            </p>

        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COPY DATA
# ============================================================

df = st.session_state.raw_data.copy()


# ============================================================
# REMOVE DUPLICATE INPUT COLUMNS
# ============================================================

df = df.loc[
    :,
    ~df.columns.duplicated(
        keep="first"
    )
].copy()


# ============================================================
# ADD REQUIRED COLUMNS IF MISSING
# ============================================================

required_numeric = [
    "Life Span (minutes)",
    "Number Of Occurrences",
    "Is Service Affecting",
    "is_active"
]

for column in required_numeric:

    if column not in df.columns:

        if column == "Number Of Occurrences":

            df[column] = 1

        else:

            df[column] = 0


# ============================================================
# TRAINING SECTION
# ============================================================

st.markdown(
    "## 🧠 Train All Models (Chained Pipeline)"
)


# ============================================================
# TRAIN BUTTON
# ============================================================

if st.button(
    "Train All Models",
    type="primary"
):

    with st.spinner(
        "Preprocessing and training all models..."
    ):

        try:

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            df_clean, df_time, encoders = (
                preprocess_alarm_data(df)
            )

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            model_dict = train_all_models(
                df_clean,
                df_time
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            save_all_models(
                model_dict,
                encoders,
                out_dir="models"
            )

            # ------------------------------------------------
            # SESSION STATE
            # ------------------------------------------------

            st.session_state.model_bundle = (
                model_dict,
                encoders
            )

            st.session_state.preprocessed_data = (
                df_clean
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            st.success(
                "✅ All models trained and saved! "
                "Ready for predictions."
            )

            # ------------------------------------------------
            # PROCESSED DATA
            # ------------------------------------------------

            st.markdown(
                "### 🧾 First rows of processed "
                "training data:"
            )

            st.dataframe(
                df_clean.head(8),
                use_container_width=True
            )

        except KeyError as error:

            st.error(
                "❌ Missing required column."
            )

            st.code(
                str(error)
            )

            st.info(
                "Available columns:"
            )

            st.write(
                list(df.columns)
            )

        except Exception as error:

            st.error(
                f"❌ Unexpected error during training: "
                f"{error}"
            )

            st.code(
                traceback.format_exc()
            )


# ============================================================
# MODEL SUMMARY
# ============================================================

if (
    "model_bundle"
    in st.session_state
):

    st.markdown(
        "## 📊 Trained Model Summary"
    )

    st.markdown(
        """
        - ✅ Alarm Type Classifier
        - ✅ Probable Cause Classifier
        - ✅ Duration Regressor
        - ✅ Severity Classifier
        - ✅ Next Alarm Timestamp Regressor
        """
    )

    st.info(
        "Switch to the Predictions page "
        "to try multi-output forecasting."
    )


# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")

col1, col2 = st.columns(
    [1, 1]
)

with col1:

    st.markdown(
        "[← Back to EDA](/eda)"
    )

with col2:

    st.markdown(
        "[Continue to Predictions →](/predictions)"
    )