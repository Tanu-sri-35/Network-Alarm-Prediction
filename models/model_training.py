import os
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# ============================================================
# BASIC HELPERS
# ============================================================

def first_series(df, column):
    """Return one Series even if duplicate column names exist."""
    value = df[column]

    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]

    return value


def remove_duplicate_columns(df):
    """Keep only the first occurrence of duplicate column names."""
    return df.loc[
        :,
        ~df.columns.duplicated(keep="first")
    ].copy()


def ensure_text(df, column, default="Unknown"):
    """Create a text column if it does not exist."""

    if column not in df.columns:
        df[column] = default

    series = first_series(
        df,
        column
    ).astype("string")

    series = series.fillna(
        default
    )

    series = series.replace(
        {
            "<NA>": default,
            "nan": default,
            "None": default
        }
    )

    df[column] = series.astype(str)

    return df


def ensure_numeric(df, column, default=0):
    """Create and clean a numeric column."""

    if column not in df.columns:
        df[column] = default

    series = pd.to_numeric(
        first_series(df, column),
        errors="coerce"
    )

    df[column] = series.fillna(
        default
    )

    return df


def create_classifier(y):
    """Create a safe classifier."""

    y = pd.Series(y)

    if y.nunique(
        dropna=False
    ) <= 1:

        return DummyClassifier(
            strategy="most_frequent"
        )

    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_alarm_data(df, cutoff=100):

    if df is None or len(df) == 0:
        raise ValueError(
            "The input dataset is empty."
        )

    df = remove_duplicate_columns(
        df.copy()
    )

    # --------------------------------------------------------
    # SAFE COLUMN MAPPING
    # --------------------------------------------------------

    aliases = {
        "alarm_type": "Alarm Type",
        "site_id": "Site Name",
        "severity": "Severity",
        "timestamp": "First Time Detected"
    }

    for old_name, new_name in aliases.items():

        if old_name in df.columns:

            if new_name not in df.columns:

                df = df.rename(
                    columns={
                        old_name: new_name
                    }
                )

            else:

                df = df.drop(
                    columns=[old_name]
                )

    # --------------------------------------------------------
    # DURATION COLUMN
    # --------------------------------------------------------

    if "duration_minutes" in df.columns:

        if "Life Span (minutes)" not in df.columns:

            df = df.rename(
                columns={
                    "duration_minutes":
                    "Life Span (minutes)"
                }
            )

        else:

            df = df.drop(
                columns=[
                    "duration_minutes"
                ]
            )

    df = remove_duplicate_columns(
        df
    )

    # --------------------------------------------------------
    # REQUIRED TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "Alarm Type",
        "Site Name",
        "Severity",
        "Source System",
        "Alarmed Object Name",
        "Alarmed Object Type",
        "Specific Problem"
    ]

    for column in text_columns:

        ensure_text(
            df,
            column
        )

    # --------------------------------------------------------
    # REQUIRED NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "Life Span (minutes)",
        "Number Of Occurrences",
        "Is Service Affecting",
        "is_active",
        "temperature"
    ]

    defaults = {
        "Life Span (minutes)": 0,
        "Number Of Occurrences": 1,
        "Is Service Affecting": 0,
        "is_active": 0,
        "temperature": 0
    }

    for column in numeric_columns:

        ensure_numeric(
            df,
            column,
            defaults[column]
        )

    # --------------------------------------------------------
    # ALARM NAME
    # --------------------------------------------------------

    if "Alarm Name" not in df.columns:

        df["Alarm Name"] = first_series(
            df,
            "Alarm Type"
        ).astype(str)

    ensure_text(
        df,
        "Alarm Name"
    )

    # --------------------------------------------------------
    # PREVIOUS SEVERITY
    # --------------------------------------------------------

    if "Previous Severity" not in df.columns:

        df["Previous Severity"] = first_series(
            df,
            "Severity"
        ).astype(str)

    ensure_text(
        df,
        "Previous Severity"
    )

    # --------------------------------------------------------
    # PROBABLE CAUSE
    # --------------------------------------------------------

    if "Probable Cause" not in df.columns:

        df["Probable Cause"] = first_series(
            df,
            "Alarm Type"
        ).astype(str)

    ensure_text(
        df,
        "Probable Cause"
    )

    # --------------------------------------------------------
    # PROBABLE CAUSE GROUPING
    # --------------------------------------------------------

    cause = first_series(
        df,
        "Probable Cause"
    ).astype(str)

    counts = cause.value_counts()

    if len(counts) == 0:

        df["Probable Cause Merged"] = "Unknown"

    else:

        common = set(
            counts[
                counts >= cutoff
            ].index
        )

        if not common:

            common = {
                counts.index[0]
            }

        df["Probable Cause Merged"] = cause.apply(
            lambda value:
            value
            if value in common
            else "Other"
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if "First Time Detected" not in df.columns:

        df["First Time Detected"] = pd.date_range(
            start="2024-01-01",
            periods=len(df),
            freq="min"
        )

    df["First Time Detected Clean"] = pd.to_datetime(
        first_series(
            df,
            "First Time Detected"
        ),
        errors="coerce"
    )

    # If all timestamps are invalid
    if (
        df["First Time Detected Clean"]
        .notna()
        .sum()
        == 0
    ):

        df["First Time Detected Clean"] = pd.date_range(
            start="2024-01-01",
            periods=len(df),
            freq="min"
        )

    df = df.dropna(
        subset=[
            "First Time Detected Clean"
        ]
    ).copy()

    df = df.sort_values(
        "First Time Detected Clean"
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    df["hour"] = (
        df["First Time Detected Clean"]
        .dt.hour
        .astype(int)
    )

    df["dayofweek"] = (
        df["First Time Detected Clean"]
        .dt.dayofweek
        .astype(int)
    )

    df["This Alarm Unix"] = (
        df["First Time Detected Clean"]
        .astype("int64")
        // 1_000_000_000
    )

    # --------------------------------------------------------
    # NEXT ALARM
    # --------------------------------------------------------

    df["Next Alarm Timestamp"] = (
        df["First Time Detected Clean"]
        .shift(-1)
    )

    df["Next Alarm Unix"] = np.nan

    valid_next = (
        df["Next Alarm Timestamp"]
        .notna()
    )

    if valid_next.any():

        df.loc[
            valid_next,
            "Next Alarm Unix"
        ] = (
            df.loc[
                valid_next,
                "Next Alarm Timestamp"
            ]
            .astype("int64")
            // 1_000_000_000
        )

    # --------------------------------------------------------
    # ENCODE CATEGORICAL COLUMNS
    # --------------------------------------------------------

    categorical_columns = [
        "Alarm Type",
        "Site Name",
        "Severity",
        "Source System",
        "Alarm Name",
        "Alarmed Object Name",
        "Alarmed Object Type",
        "Specific Problem",
        "Previous Severity",
        "Probable Cause Merged"
    ]

    encoders = {}

    for column in categorical_columns:

        ensure_text(
            df,
            column
        )

        encoder = LabelEncoder()

        values = first_series(
            df,
            column
        ).astype(str)

        df[column] = encoder.fit_transform(
            values
        )

        encoders[column] = encoder

    # --------------------------------------------------------
    # FINAL NUMERIC CLEANUP
    # --------------------------------------------------------

    for column in [
        "Life Span (minutes)",
        "Number Of Occurrences",
        "Is Service Affecting",
        "is_active",
        "temperature",
        "hour",
        "dayofweek"
    ]:

        ensure_numeric(
            df,
            column,
            0
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = remove_duplicate_columns(
        df
    )

    # --------------------------------------------------------
    # DATA FOR NEXT ALARM MODEL
    # --------------------------------------------------------

    df_time = df[
        df["Next Alarm Unix"].notna()
    ].copy()

    if not df_time.empty:

        df_time = df_time[
            df_time["Next Alarm Unix"]
            > df_time["This Alarm Unix"]
        ].copy()

    df_time = remove_duplicate_columns(
        df_time
    )

    # --------------------------------------------------------
    # REMOVE UNNECESSARY COLUMNS
    # --------------------------------------------------------

    columns_to_remove = [
        "First Time Detected",
        "Last Time Cleared",
        "Last Time Detected",
        "Additional Text",
        "Probable Cause",
        "Next Alarm Timestamp"
    ]

    df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    df = remove_duplicate_columns(
        df
    )

    return (
        df,
        df_time,
        encoders
    )


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models(
    df,
    df_time
):

    df = remove_duplicate_columns(
        df.copy()
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    alarm_type_features = [
        "Site Name",
        "Severity",
        "Source System",
        "Alarmed Object Type",
        "Previous Severity",
        "Is Service Affecting",
        "Number Of Occurrences",
        "is_active",
        "temperature",
        "hour",
        "dayofweek"
    ]

    cause_features = (
        alarm_type_features
        + [
            "Alarm Type"
        ]
    )

    duration_features = (
        cause_features
        + [
            "Probable Cause Merged"
        ]
    )

    severity_features = (
        duration_features
        + [
            "Life Span (minutes)"
        ]
    )

    timestamp_features = [
        "Severity",
        "Site Name",
        "Source System",
        "Probable Cause Merged",
        "Alarmed Object Type",
        "Previous Severity",
        "Is Service Affecting",
        "Number Of Occurrences",
        "is_active",
        "Alarm Type",
        "Life Span (minutes)",
        "temperature",
        "hour",
        "dayofweek"
    ]

    all_features = set(
        alarm_type_features
        + cause_features
        + duration_features
        + severity_features
        + timestamp_features
    )

    # --------------------------------------------------------
    # ENSURE FEATURES EXIST
    # --------------------------------------------------------

    for column in all_features:

        if column not in df.columns:

            df[column] = 0

        df[column] = pd.to_numeric(
            first_series(
                df,
                column
            ),
            errors="coerce"
        ).fillna(0)

    df = remove_duplicate_columns(
        df
    )

    # ========================================================
    # MODEL 1 — ALARM TYPE
    # ========================================================

    X_alarm = df[
        alarm_type_features
    ].copy()

    y_alarm = first_series(
        df,
        "Alarm Type"
    )

    alarm_type_model = create_classifier(
        y_alarm
    )

    alarm_type_model.fit(
        X_alarm,
        y_alarm
    )

    # ========================================================
    # MODEL 2 — PROBABLE CAUSE
    # ========================================================

    X_cause = df[
        cause_features
    ].copy()

    y_cause = first_series(
        df,
        "Probable Cause Merged"
    )

    cause_model = create_classifier(
        y_cause
    )

    cause_model.fit(
        X_cause,
        y_cause
    )

    # ========================================================
    # MODEL 3 — DURATION
    # ========================================================

    X_duration = df[
        duration_features
    ].copy()

    y_duration = pd.to_numeric(
        first_series(
            df,
            "Life Span (minutes)"
        ),
        errors="coerce"
    ).fillna(0)

    duration_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    duration_model.fit(
        X_duration,
        y_duration
    )

    # ========================================================
    # MODEL 4 — SEVERITY
    # ========================================================

    X_severity = df[
        severity_features
    ].copy()

    y_severity = first_series(
        df,
        "Previous Severity"
    )

    severity_model = create_classifier(
        y_severity
    )

    severity_model.fit(
        X_severity,
        y_severity
    )

    # ========================================================
    # MODEL 5 — NEXT ALARM TIMESTAMP
    # ========================================================

    timestamp_model = None

    if (
        df_time is not None
        and len(df_time) >= 2
    ):

        df_time = remove_duplicate_columns(
            df_time.copy()
        )

        for column in timestamp_features:

            if column not in df_time.columns:

                df_time[column] = 0

            df_time[column] = pd.to_numeric(
                first_series(
                    df_time,
                    column
                ),
                errors="coerce"
            ).fillna(0)

        X_time = df_time[
            timestamp_features
        ].copy()

        y_time = pd.to_numeric(
            first_series(
                df_time,
                "Next Alarm Unix"
            ),
            errors="coerce"
        )

        valid_rows = y_time.notna()

        X_time = X_time.loc[
            valid_rows
        ].copy()

        y_time = y_time.loc[
            valid_rows
        ].copy()

        if len(X_time) >= 2:

            timestamp_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            timestamp_model.fit(
                X_time,
                y_time
            )

    # ========================================================
    # RETURN MODELS
    # ========================================================

    return {
        "alarm_type_clf": alarm_type_model,

        "cause_clf": cause_model,

        "duration_reg": duration_model,

        "severity_clf": severity_model,

        "timestamp_reg": timestamp_model,

        "input_features": {
            "alarm_type": alarm_type_features,
            "cause": cause_features,
            "duration": duration_features,
            "severity": severity_features,
            "timestamp": timestamp_features
        }
    }


# ============================================================
# SAVE MODELS
# ============================================================

def save_all_models(
    model_dict,
    encoders,
    out_dir="models"
):

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    model_names = [
        "alarm_type_clf",
        "cause_clf",
        "duration_reg",
        "severity_clf",
        "timestamp_reg"
    ]

    for name in model_names:

        model = model_dict.get(
            name
        )

        if model is not None:

            path = os.path.join(
                out_dir,
                name + ".joblib"
            )

            joblib.dump(
                model,
                path
            )

    joblib.dump(
        encoders,
        os.path.join(
            out_dir,
            "encoders.joblib"
        )
    )

    joblib.dump(
        model_dict.get(
            "input_features",
            {}
        ),
        os.path.join(
            out_dir,
            "input_features.joblib"
        )
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_all_models(
    out_dir="models"
):

    model_dict = {}

    model_names = [
        "alarm_type_clf",
        "cause_clf",
        "duration_reg",
        "severity_clf",
        "timestamp_reg"
    ]

    for name in model_names:

        path = os.path.join(
            out_dir,
            name + ".joblib"
        )

        if os.path.exists(path):

            model_dict[name] = joblib.load(
                path
            )

        else:

            model_dict[name] = None

    feature_path = os.path.join(
        out_dir,
        "input_features.joblib"
    )

    if os.path.exists(
        feature_path
    ):

        model_dict["input_features"] = joblib.load(
            feature_path
        )

    else:

        model_dict["input_features"] = {}

    encoder_path = os.path.join(
        out_dir,
        "encoders.joblib"
    )

    if os.path.exists(
        encoder_path
    ):

        encoders = joblib.load(
            encoder_path
        )

    else:

        encoders = {}

    return (
        model_dict,
        encoders
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

def chained_predict(
    row,
    model_dict,
    encoders
):

    if isinstance(
        row,
        pd.Series
    ):

        row = row.to_frame().T

    row = remove_duplicate_columns(
        row.copy()
    )

    features = model_dict.get(
        "input_features",
        {}
    )

    # Add missing columns
    for group in features.values():

        for column in group:

            if column not in row.columns:

                row[column] = 0

    # Convert all model inputs to numbers
    for group in features.values():

        for column in group:

            row[column] = pd.to_numeric(
                first_series(
                    row,
                    column
                ),
                errors="coerce"
            ).fillna(0)

    # --------------------------------------------------------
    # ALARM TYPE
    # --------------------------------------------------------

    X_alarm = row[
        features["alarm_type"]
    ].copy()

    alarm_prediction = (
        model_dict[
            "alarm_type_clf"
        ]
        .predict(
            X_alarm
        )[0]
    )

    # --------------------------------------------------------
    # CAUSE
    # --------------------------------------------------------

    cause_row = row.copy()

    cause_row[
        "Alarm Type"
    ] = alarm_prediction

    X_cause = cause_row[
        features["cause"]
    ].copy()

    cause_prediction = (
        model_dict[
            "cause_clf"
        ]
        .predict(
            X_cause
        )[0]
    )

    # --------------------------------------------------------
    # DURATION
    # --------------------------------------------------------

    duration_row = cause_row.copy()

    duration_row[
        "Probable Cause Merged"
    ] = cause_prediction

    X_duration = duration_row[
        features["duration"]
    ].copy()

    duration_prediction = (
        model_dict[
            "duration_reg"
        ]
        .predict(
            X_duration
        )[0]
    )

    duration_prediction = max(
        0.0,
        float(duration_prediction)
    )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    severity_row = duration_row.copy()

    severity_row[
        "Life Span (minutes)"
    ] = duration_prediction

    X_severity = severity_row[
        features["severity"]
    ].copy()

    severity_prediction = (
        model_dict[
            "severity_clf"
        ]
        .predict(
            X_severity
        )[0]
    )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    def decode(
        encoder_name,
        value
    ):

        encoder = encoders.get(
            encoder_name
        )

        if encoder is None:

            return str(value)

        try:

            return encoder.inverse_transform(
                [int(value)]
            )[0]

        except Exception:

            return str(value)

    return {
        "Predicted Alarm Type": decode(
            "Alarm Type",
            alarm_prediction
        ),

        "Predicted Probable Cause": decode(
            "Probable Cause Merged",
            cause_prediction
        ),

        "Predicted Duration (minutes)": (
            duration_prediction
        ),

        "Predicted Severity": decode(
            "Previous Severity",
            severity_prediction
        )
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

def batch_chained_predict(
    X,
    model_dict,
    encoders
):

    results = []

    if isinstance(
        X,
        pd.Series
    ):

        X = X.to_frame().T

    for index, row in X.iterrows():

        try:

            result = chained_predict(
                row,
                model_dict,
                encoders
            )

            result["row"] = index

            results.append(
                result
            )

        except Exception as error:

            results.append(
                {
                    "row": index,
                    "Prediction Error": str(error)
                }
            )

    return pd.DataFrame(
        results
    )