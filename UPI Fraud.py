# fraud_sight_app.py
import os
import warnings

warnings.filterwarnings("ignore")

# ----------------- Core Imports -----------------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
plt.rcParams["figure.figsize"] = (3,2)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

import joblib

# Try optional XGBoost
HAS_XGB = False
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- Config -----------------
DEFAULT_CSV = r"C:/Users/Sushant raj/Downloads/PS_20174392719_1491204439457_log.csv"
MODEL_DIR = "models_portfolio"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Fraud-Sight — Hybrid UPI Fraud Detection", layout="wide"
)
st.title("Fraud-Sight — Hybrid UPI Fraud Detection")


# ----------------- Helpers -----------------


@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)


def basic_eda_info(df: pd.DataFrame):
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return info


def preprocess_for_models(df, features=None, target="isFraud"):
    """Prepare X, y, encoders and scaler."""
    df = df.copy()

    if target not in df.columns:
        return None, None, None, None, None, None

    # Drop rows without target
    df = df.dropna(subset=[target])

    # Default feature set
    if features is None:
        default_order = [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFlaggedFraud",
        ]
        features = [c for c in default_order if c in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            if c not in features and c != target:
                features.append(c)

    encoders = {}
    df_features = df[features].copy()

    # Fill & encode
    for c in features:
        if df_features[c].dtype == "object" or df_features[c].dtype.name == "category":
            df_features[c] = df_features[c].fillna("NA").astype(str)
            le = LabelEncoder()
            df_features[c] = le.fit_transform(df_features[c])
            encoders[c] = le
        else:
            df_features[c] = df_features[c].fillna(df_features[c].median())

    X = df_features
    y = df[target].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, X, y, features, encoders, scaler


def evaluate_model(y_true, y_pred, y_proba=None):
    """Calculates accuracy, precision, recall, F1, ROC-AUC, PR-AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_val = None
    pr_auc_val = None

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_val = roc_auc_score(y_true, y_proba)
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
            pr_auc_val = auc(rec_curve, prec_curve)
        except Exception:
            pass

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_val,
        "pr_auc": pr_auc_val,
    }


def confusion_matrix_fig(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ----------------- Sidebar: Data & Controls -----------------
st.sidebar.header("Data & Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload UPI fraud CSV (optional)", type=["csv"]
)
use_default = st.sidebar.checkbox("Use default CSV path", value=True)
csv_path = st.sidebar.text_input("CSV local path", value=DEFAULT_CSV)

with st.sidebar.expander("Training & Hyperparameters", expanded=False):
    rf_n_estimators = st.number_input(
        "RandomForest n_estimators", min_value=50, max_value=500, value=200, step=50
    )
    iso_contamination = st.number_input(
        "IsolationForest contamination",
        min_value=0.000,
        max_value=0.05,
        value=0.002,
        step=0.001,
    )
    test_size = st.slider(
        "Test size fraction", min_value=0.05, max_value=0.5, value=0.25
    )
    train_button = st.button("Train / Retrain Models")

st.sidebar.markdown("---")
st.sidebar.write(f"Models & artifacts saved in: `{MODEL_DIR}/`")

# ----------------- Load Data -----------------
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    if use_default:
        try:
            df = load_csv(csv_path)
        except Exception as e:
            st.sidebar.error(f"Could not read default path: {e}")
            st.stop()
    else:
        st.info("Upload a CSV or enable default path in the sidebar.")
        st.stop()

st.sidebar.markdown(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
if st.sidebar.checkbox("Show sample data (first 200 rows)"):
    st.dataframe(df.head(200), use_container_width=True)

# ----------------- Session State -----------------
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False

if "results" not in st.session_state:
    st.session_state.results = None

if "preprocess" not in st.session_state:
    st.session_state.preprocess = None  # holds features, encoders, scaler

if "splits" not in st.session_state:
    st.session_state.splits = None  # holds X_train, X_test, y_train, y_test


# ----------------- Training Pipeline -----------------
def train_all_models(
    df_in, rf_n=200, iso_cont=0.002, test_size=0.25, target="isFraud"
):
    X_scaled, X_raw, y, features, encoders, scaler = preprocess_for_models(
        df_in, target=target
    )

    if X_scaled is None:
        st.error(f"Target column '{target}' not found. Supervised training aborted.")
        return None

    X_train, X_test, y_train, y_test, X_train_df, X_test_df = train_test_split(
        X_scaled,
        y,
        X_raw,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = {}
    models = {}

    # -------- Rule-Based (approx) – very shallow RF as baseline --------
    rule_rf = RandomForestClassifier(
        n_estimators=1, max_depth=1, random_state=RANDOM_STATE
    )
    rule_rf.fit(X_train, y_train)
    rb_pred = rule_rf.predict(X_test)
    rb_proba = rule_rf.predict_proba(X_test)[:, 1]
    metrics_rb = evaluate_model(y_test, rb_pred, rb_proba)
    models["Rule-Based"] = rule_rf
    results["rule_based"] = {
        "name": "Rule-Based",
        "metrics": metrics_rb,
        "y_test": y_test,
        "y_pred": rb_pred,
        "y_proba": rb_proba,
    }

    # -------- Random Forest --------
    rf = RandomForestClassifier(
        n_estimators=rf_n, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    metrics_rf = evaluate_model(y_test, rf_pred, rf_proba)
    models["Random Forest"] = rf
    results["rf"] = {
        "name": "Random Forest",
        "metrics": metrics_rf,
        "y_test": y_test,
        "y_pred": rf_pred,
        "y_proba": rf_proba,
    }

    # -------- XGBoost (if available) --------
    if HAS_XGB:
        xgb = XGBClassifier(
            eval_metric="logloss",
            n_estimators=rf_n,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        metrics_xgb = evaluate_model(y_test, xgb_pred, xgb_proba)
        models["XGBoost"] = xgb
        results["xgb"] = {
            "name": "XGBoost",
            "metrics": metrics_xgb,
            "y_test": y_test,
            "y_pred": xgb_pred,
            "y_proba": xgb_proba,
        }

    # -------- Isolation Forest (unsupervised) --------
    nonfraud_mask = y_train == 0
    if nonfraud_mask.sum() < 10:
        iso_train_X = X_train
    else:
        iso_train_X = X_train[nonfraud_mask]

    iso = IsolationForest(
        contamination=float(iso_cont), random_state=RANDOM_STATE
    )
    iso.fit(iso_train_X)
    iso_pred_raw = iso.predict(X_test)  # -1 anomaly, 1 normal
    iso_flag = np.where(iso_pred_raw == -1, 1, 0)
    metrics_iso = evaluate_model(y_test, iso_flag, None)
    models["Isolation Forest"] = iso
    results["iso"] = {
        "name": "IsolationForest",
        "metrics": metrics_iso,
        "y_test": y_test,
        "y_pred": iso_flag,
        "y_proba": None,
    }

    # -------- Hybrid Model: RF_proba + ISO score -> LogisticRegression --------
    rf_proba_train = rf.predict_proba(X_train)[:, 1]
    rf_proba_test = rf_proba

    iso_scores_train = iso.score_samples(X_train)
    iso_scores_test = iso.score_samples(X_test)

    X_meta_train = np.vstack([rf_proba_train, iso_scores_train]).T
    X_meta_test = np.vstack([rf_proba_test, iso_scores_test]).T

    meta_scaler = StandardScaler()
    X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
    X_meta_test_scaled = meta_scaler.transform(X_meta_test)

    meta_clf = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    )
    meta_clf.fit(X_meta_train_scaled, y_train)
    meta_pred = meta_clf.predict(X_meta_test_scaled)
    meta_proba = meta_clf.predict_proba(X_meta_test_scaled)[:, 1]
    metrics_meta = evaluate_model(y_test, meta_pred, meta_proba)
    models["Hybrid Model"] = meta_clf
    results["meta"] = {
        "name": "Hybrid Model",
        "metrics": metrics_meta,
        "y_test": y_test,
        "y_pred": meta_pred,
        "y_proba": meta_proba,
        "meta_scaler": meta_scaler,
    }

    # -------- Save artifacts --------
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso_model.joblib"))
    joblib.dump(meta_clf, os.path.join(MODEL_DIR, "hybrid_model.joblib"))
    joblib.dump(meta_scaler, os.path.join(MODEL_DIR, "hybrid_meta_scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    st.session_state.preprocess = {
        "features": features,
        "encoders": encoders,
        "scaler": scaler,
    }
    st.session_state.splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_df": X_test_df,
    }

    return results


def build_comparison_table(results_dict):
    rows = []
    name_map_order = [
        "rule_based",
        "rf",
        "xgb",
        "iso",
        "meta",
    ]
    for key in name_map_order:
        if key in results_dict:
            m = results_dict[key]["metrics"]
            rows.append(
                {
                    "Model": results_dict[key]["name"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1-Score": m["f1"],
                    "ROC-AUC": m["roc_auc"],
                    "PR-AUC": m["pr_auc"],
                    "Accuracy": m["accuracy"],
                }
            )
    return pd.DataFrame(rows)


# ----------------- Trigger training -----------------
if train_button:
    with st.spinner("Training models on dataset..."):
        res = train_all_models(
            df,
            rf_n=rf_n_estimators,
            iso_cont=iso_contamination,
            test_size=test_size,
            target="isFraud",
        )
        if res is not None:
            st.session_state.results = res
            st.session_state.models_trained = True
            st.success("Models trained and hybrid model built successfully!")
        else:
            st.error("Training failed. Check logs / target column.")


results = st.session_state.results

# ----------------- Page Navigation -----------------
page = st.sidebar.radio(
    "Navigate",
    ["Home", "EDA", "Model Training & Comparison", "Hybrid Performance", "Prediction Tool", "Bulk Scoring", "About"],
)

# ==========================
# PAGE: HOME
# ==========================
if page == "Home":
    st.header("Overview & Key Metrics")
    st.write(
        "This app demonstrates a **hybrid machine learning approach** to detect UPI fraud, "
        "combining supervised models (Random Forest, XGBoost) and an unsupervised model (IsolationForest). "
        "A meta-model then fuses these signals into a stronger hybrid classifier."
    )

    info = basic_eda_info(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", info["rows"])
    c2.metric("Columns", info["columns"])
    c3.metric("Missing values", info["missing_values"])
    c4.metric("Duplicate rows", info["duplicate_rows"])

    if results is not None:
        st.subheader("Model Comparison Table")
        comp_df = build_comparison_table(results)
        st.dataframe(comp_df, use_container_width=True)
        st.download_button(
            "Download model comparison (CSV)",
            data=df_to_csv_bytes(comp_df),
            file_name="model_comparison.csv",
        )

        if "meta" in results:
            st.markdown("### Hybrid Model – Key Metrics")
            m = results["meta"]["metrics"]
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Precision", f"{m['precision']:.4f}")
            k2.metric("Recall", f"{m['recall']:.4f}")
            k3.metric("F1-score", f"{m['f1']:.4f}")
            k4.metric(
                "ROC-AUC", f"{m['roc_auc']:.4f}"
                if m["roc_auc"] is not None
                else "N/A"
            )
            k5.metric("Accuracy", f"{m['accuracy']:.4f}")

# ==========================
# PAGE: EDA
# ==========================
elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("1. Basic Info")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Columns:", list(df.columns))

    st.subheader("2. Target Variable Distribution (isFraud)")
    if "isFraud" in df.columns:
        fig, ax = plt.subplots()
        df["isFraud"].value_counts().plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_xticklabels(["Non-fraud (0)", "Fraud (1)"])
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("3. Transaction Amount Distribution")
    if "amount" in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(
            df["amount"].replace(0, np.nan).dropna(),
            bins=80,
            kde=True,
            ax=ax2,
        )
        ax2.set_xscale("log")
        ax2.set_title("Transaction Amount (log scale)")
        st.pyplot(fig2)

    st.subheader("4. Fraud by Transaction Type")
    if "type" in df.columns and "isFraud" in df.columns:
        fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots()
        sns.barplot(
            x=fraud_by_type.index,
            y=fraud_by_type.values,
            ax=ax3,
        )
        ax3.set_ylabel("Fraud Rate")
        ax3.set_xlabel("Transaction Type")
        ax3.set_title("Average Fraud Rate by Type")
        st.pyplot(fig3)

    st.subheader("5. Correlation Heatmap (numeric features)")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), annot=False, cmap="coolwarm", ax=ax4)
        ax4.set_title("Correlation Matrix")
        st.pyplot(fig4)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# ==========================
# PAGE: Model Training & Comparison
# ==========================
elif page == "Model Training & Comparison":
    st.header("Model Training & Comparison")

    if results is None:
        st.info("Click **Train / Retrain Models** in the sidebar to train.")
    else:
        st.subheader("Comparison Table")
        comp_df = build_comparison_table(results)
        st.dataframe(
    comp_df.style.format(
        {col: "{:.4f}" for col in comp_df.columns if comp_df[col].dtype != "object"}
    ),
    width='stretch'
)


        st.subheader("Confusion Matrices")
        cols = st.columns(3)

        if "rule_based" in results:
            fig_rb = confusion_matrix_fig(
                results["rule_based"]["y_test"],
                results["rule_based"]["y_pred"],
                "Rule-Based",
            )
            cols[0].pyplot(fig_rb)

        if "rf" in results:
            fig_rf = confusion_matrix_fig(
                results["rf"]["y_test"],
                results["rf"]["y_pred"],
                "Random Forest",
            )
            cols[1].pyplot(fig_rf)

        if "xgb" in results:
            fig_xgb = confusion_matrix_fig(
                results["xgb"]["y_test"],
                results["xgb"]["y_pred"],
                "XGBoost",
            )
            cols[2].pyplot(fig_xgb)

        st.subheader("IsolationForest Confusion Matrix")
        if "iso" in results:
            fig_iso = confusion_matrix_fig(
                results["iso"]["y_test"],
                results["iso"]["y_pred"],
                "IsolationForest",
            )
            st.pyplot(fig_iso)

# ==========================
# PAGE: Hybrid Performance
# ==========================
elif page == "Hybrid Performance":
    st.header("Hybrid Model: Performance & Curves")

    if results is None or "meta" not in results:
        st.info("Train models first to see hybrid performance.")
    else:
        meta_res = results["meta"]
        y_test = meta_res["y_test"]
        y_proba = meta_res["y_proba"]

        st.subheader("Hybrid vs Random Forest — ROC & PR Curves")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        if "rf" in results:
            fpr_rf, tpr_rf, _ = roc_curve(results["rf"]["y_test"], results["rf"]["y_proba"])
            ax[0].plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc(fpr_rf, tpr_rf):.3f})")

        fpr_m, tpr_m, _ = roc_curve(y_test, y_proba)
        ax[0].plot(fpr_m, tpr_m, label=f"Hybrid (AUC={auc(fpr_m, tpr_m):.3f})")
        ax[0].plot([0, 1], [0, 1], "k--")
        ax[0].set_title("ROC Curve")
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].legend()

        # PR
        if "rf" in results:
            prec_rf, rec_rf, _ = precision_recall_curve(
                results["rf"]["y_test"], results["rf"]["y_proba"]
            )
            ax[1].plot(rec_rf, prec_rf, label=f"RF (PR-AUC={auc(rec_rf, prec_rf):.3f})")

        prec_m, rec_m, _ = precision_recall_curve(y_test, y_proba)
        ax[1].plot(rec_m, prec_m, label=f"Hybrid (PR-AUC={auc(rec_m, prec_m):.3f})")
        ax[1].set_title("Precision-Recall Curve")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].legend()

        st.pyplot(fig)

        st.subheader("Random Forest Feature Importances")
        if "rf" in results and st.session_state.preprocess is not None:
            rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
            features = st.session_state.preprocess["features"]
            fi = pd.Series(rf_model.feature_importances_, index=features).sort_values(
                ascending=False
            )
            fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
            sns.barplot(x=fi.values, y=fi.index, ax=ax_fi)
            ax_fi.set_title("RF Feature Importances")
            st.pyplot(fig_fi)

# ==========================
# PAGE: Prediction Tool
# ==========================
elif page == "Prediction Tool":
    st.header("Single Transaction Prediction")

    if results is None or st.session_state.preprocess is None:
        st.info("Train models first to enable prediction.")
    else:
        features = st.session_state.preprocess["features"]
        encoders = st.session_state.preprocess["encoders"]
        scaler = st.session_state.preprocess["scaler"]

        rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
        iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))
        meta_model = joblib.load(os.path.join(MODEL_DIR, "hybrid_model.joblib"))
        meta_scaler = joblib.load(os.path.join(MODEL_DIR, "hybrid_meta_scaler.joblib"))

        st.markdown("Fill in the transaction details to score fraud risk:")

        with st.form("single_tx_form"):
            tx_dict = {}
            for f in features:
                if f in df.columns and (
                    df[f].dtype == "object" or df[f].dtype.name == "category"
                ):
                    default_val = (
                        str(df[f].mode().iloc[0]) if not df[f].mode().empty else ""
                    )
                    tx_dict[f] = st.text_input(f, value=default_val)
                elif f in df.columns:
                    median_val = float(df[f].median()) if not df[f].median() is None else 0.0
                    tx_dict[f] = st.number_input(f, value=median_val, format="%.4f")
                else:
                    tx_dict[f] = st.text_input(f, value="")
            submitted = st.form_submit_button("Predict Fraud Risk")

        if submitted:
            row = pd.DataFrame([tx_dict], columns=features)

            # Encode categoricals
            for col, le in encoders.items():
                if col in row.columns:
                    val = str(row[col].iloc[0])
                    if val in le.classes_:
                        row[col] = le.transform([val])
                    else:
                        row[col] = -1

            # Numeric fill
            for col in row.columns:
                if col not in encoders:
                    row[col] = pd.to_numeric(row[col], errors="coerce")
                    if pd.isna(row[col].iloc[0]) and col in df.columns:
                        row[col] = df[col].median()
                    elif pd.isna(row[col].iloc[0]):
                        row[col] = 0.0

            X_row = scaler.transform(row.values.astype(float))

            rf_prob = rf_model.predict_proba(X_row)[:, 1][0]
            iso_score = iso_model.score_samples(X_row)[0]
            meta_input = meta_scaler.transform([[rf_prob, iso_score]])
            hybrid_prob = meta_model.predict_proba(meta_input)[:, 1][0]
            label = "Fraud" if hybrid_prob >= 0.5 else "Genuine"

            c1, c2, c3 = st.columns(3)
            c1.metric("RF probability", f"{rf_prob:.4f}")
            c2.metric("ISO score (higher = more normal)", f"{iso_score:.4f}")
            c3.metric("Hybrid fraud probability", f"{hybrid_prob:.4f}")
            st.write(f"**Final predicted label:** `{label}`")

# ==========================
# PAGE: Bulk Scoring
# ==========================
elif page == "Bulk Scoring":
    st.header("Bulk Fraud Detection — CSV Scoring")

    if results is None or st.session_state.preprocess is None:
        st.info("Train models first to enable bulk scoring.")
    else:
        st.write(
            "Upload a CSV with the same schema as the training data (transaction rows)."
        )
        bulk_file = st.file_uploader("Upload CSV for scoring", type=["csv"])

        if bulk_file is not None:
            try:
                bulk_df = pd.read_csv(bulk_file)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                bulk_df = None
        else:
            bulk_df = None

        if bulk_df is not None:
            st.write("Preview of data to be scored:")
            st.dataframe(bulk_df.head(100), use_container_width=True)

            features = st.session_state.preprocess["features"]
            encoders = st.session_state.preprocess["encoders"]
            scaler = st.session_state.preprocess["scaler"]

            rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
            iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))
            meta_model = joblib.load(os.path.join(MODEL_DIR, "hybrid_model.joblib"))
            meta_scaler = joblib.load(os.path.join(MODEL_DIR, "hybrid_meta_scaler.joblib"))

            run_scoring = st.button("Run Bulk Scoring")

            if run_scoring:
                use_df = bulk_df.copy()

                # Ensure all features exist
                for f in features:
                    if f not in use_df.columns:
                        use_df[f] = np.nan

                use_df = use_df[features].copy()

                # Encode
                for col, le in encoders.items():
                    if col in use_df.columns:
                        use_df[col] = use_df[col].astype(str)
                        use_df[col] = use_df[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )

                # Fill numeric
                for col in use_df.columns:
                    if col not in encoders:
                        use_df[col] = pd.to_numeric(use_df[col], errors="coerce")
                        use_df[col] = use_df[col].fillna(use_df[col].median())

                X_bulk = scaler.transform(use_df.values.astype(float))

                rf_prob = rf_model.predict_proba(X_bulk)[:, 1]
                iso_score = iso_model.score_samples(X_bulk)
                meta_input = meta_scaler.transform(
                    np.vstack([rf_prob, iso_score]).T
                )
                hybrid_prob = meta_model.predict_proba(meta_input)[:, 1]
                labels = (hybrid_prob >= 0.5).astype(int)

                scored_df = bulk_df.copy()
                scored_df["rf_prob"] = rf_prob
                scored_df["iso_score"] = iso_score
                scored_df["hybrid_prob"] = hybrid_prob
                scored_df["predicted_label"] = labels

                st.write("Scored sample (first 200 rows):")
                st.dataframe(scored_df.head(200), use_container_width=True)

                st.download_button(
                    "Download scored CSV",
                    data=df_to_csv_bytes(scored_df),
                    file_name="upi_scored_transactions.csv",
                )

# ==========================
# PAGE: About
# ==========================
else:
    st.header("About This Project")
    st.markdown(
        """
### Combating UPI Fraud with a Hybrid ML Approach

This portfolio project demonstrates:

- **Real-world dataset**: UPI-like transactional data with `isFraud` labels  
- **Supervised models**: RandomForest, optional XGBoost  
- **Unsupervised model**: IsolationForest for anomaly detection  
- **Hybrid meta-model**: Logistic Regression trained on RF probability + ISO anomaly score  
- **Interactive dashboard** built with Streamlit  
- **EDA section** for understanding fraud patterns  
- **Single transaction scoring** + **bulk CSV fraud detection**  

You can extend this by:
- Adding SHAP-based explainability
- Trying other ensemble models (LightGBM, CatBoost)
- Tuning hyperparameters with GridSearchCV / Optuna
- Deploying the app on Streamlit Cloud, Render, or HuggingFace Spaces
"""
    )

    st.subheader("Quick Dataset Glimpse")
    st.dataframe(df.head(50), use_container_width=True)
