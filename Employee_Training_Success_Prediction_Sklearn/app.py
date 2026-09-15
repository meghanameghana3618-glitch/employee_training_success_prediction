import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Employee Training Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme and Modern Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #064E3B 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.3);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-bottom: 10px;
    }

    .result-card-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .result-card-improvement {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 6px 0;
    }

    .recommendation-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find model & dataset
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Employee_Training_Success_Prediction_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("employee_training_success_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading training success model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">L&D Intelligence & Upskilling Analytics</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🎓 Employee Training Success Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast employee upskilling outcomes (Successful vs Needs Improvement) using machine learning based on pre-training aptitude, attendance, and professional background.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Trainee Evaluation", "📁 Batch Cohort Analysis (CSV)", "📊 Model Performance & Insights"])

# --- TAB 1: Trainee Evaluation ---
with tabs[0]:
    st.subheader("Trainee Profile & Learning Metrics")

    # Quick Preset Buttons
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        preset_top = st.button("🌟 High-Potential Trainee", width="stretch")
    with p_cols[1]:
        preset_risk = st.button("⚠️ At-Risk Trainee", width="stretch")
    with p_cols[2]:
        preset_sample = st.button("📋 Sample Profile", width="stretch")

    if preset_top:
        st.session_state["age"] = 28
        st.session_state["exp"] = 5
        st.session_state["dept"] = "IT"
        st.session_state["hours"] = 55
        st.session_state["attendance"] = 96
        st.session_state["pre_score"] = 82
    elif preset_risk:
        st.session_state["age"] = 54
        st.session_state["exp"] = 22
        st.session_state["dept"] = "Operations"
        st.session_state["hours"] = 18
        st.session_state["attendance"] = 58
        st.session_state["pre_score"] = 38
    elif preset_sample:
        st.session_state["age"] = 29
        st.session_state["exp"] = 4
        st.session_state["dept"] = "IT"
        st.session_state["hours"] = 45
        st.session_state["attendance"] = 92
        st.session_state["pre_score"] = 72

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 👤 Employee Background")
        age = st.slider(
            "Age (Years)", 20, 65,
            value=int(st.session_state.get("age", 29)),
            key="input_age"
        )
        years_experience = st.slider(
            "Years of Professional Experience", 0, 30,
            value=int(st.session_state.get("exp", 4)),
            key="input_exp"
        )
        department = st.selectbox(
            "Department",
            ["IT", "HR", "Operations", "Sales", "Finance"],
            index=["IT", "HR", "Operations", "Sales", "Finance"].index(st.session_state.get("dept", "IT")),
            key="input_dept"
        )

    with c_right:
        st.markdown("#### 📚 Training Commitment & Aptitude")
        training_hours = st.slider(
            "Training Program Hours", 10, 80,
            value=int(st.session_state.get("hours", 45)),
            key="input_hours",
            help="Total dedicated training hours completed."
        )
        attendance_percent = st.slider(
            "Session Attendance (%)", 50, 100,
            value=int(st.session_state.get("attendance", 92)),
            key="input_attendance",
            help="Percentage of scheduled training sessions attended."
        )
        pre_training_score = st.slider(
            "Pre-Training Assessment Score (0 - 100)", 20, 100,
            value=int(st.session_state.get("pre_score", 72)),
            key="input_pre_score",
            help="Baseline knowledge benchmark score achieved prior to training."
        )

    st.markdown("---")
    predict_btn = st.button("🚀 Evaluate Training Outcome", type="primary", width="stretch")

    sample_df = pd.DataFrame([{
        "age": age,
        "years_experience": years_experience,
        "training_hours": training_hours,
        "attendance_percent": attendance_percent,
        "pre_training_score": pre_training_score,
        "department": department
    }])

    # Run Prediction
    prediction = model.predict(sample_df)[0]
    probabilities = model.predict_proba(sample_df)[0]
    classes = list(model.classes_)
    pred_conf = max(probabilities) * 100

    # Probability of Successful
    success_idx = classes.index("Successful") if "Successful" in classes else 1
    success_prob = probabilities[success_idx] * 100

    st.markdown("### 📋 Evaluation Result")
    r_col1, r_col2 = st.columns([1.3, 1.7], gap="medium")

    with r_col1:
        if prediction == "Successful":
            st.markdown(f"""
            <div class="result-card-success">
                <span style="font-size: 2.8rem;">🎉</span>
                <div style="color: #34D399; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    Successful Completion
                </div>
                <div class="metric-value" style="color: #10B981;">
                    {pred_conf:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Model Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-improvement">
                <span style="font-size: 2.8rem;">⚠️</span>
                <div style="color: #F87171; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    Needs Improvement
                </div>
                <div class="metric-value" style="color: #EF4444;">
                    {pred_conf:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Model Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        st.write(f"**Success Likelihood:** {success_prob:.1f}%")
        st.progress(float(success_prob / 100.0))

    with r_col2:
        st.markdown("#### 🔍 Performance Drivers & Learning Diagnostics")
        insights = []
        if attendance_percent >= 90:
            insights.append(("High Attendance Commitment", f"{attendance_percent}% attendance strongly drives skill retention.", "positive"))
        elif attendance_percent < 70:
            insights.append(("Attendance Deficit", f"{attendance_percent}% attendance is a primary risk factor for incomplete mastery.", "negative"))

        if pre_training_score >= 70:
            insights.append(("Strong Baseline Aptitude", f"Score of {pre_training_score} indicates solid prerequisite foundations.", "positive"))
        elif pre_training_score < 50:
            insights.append(("Low Baseline Knowledge", f"Pre-training score of {pre_training_score} suggests difficulty with core concepts.", "negative"))

        if training_hours >= 40:
            insights.append(("Sufficient Practice Duration", f"{training_hours} training hours provides ample hands-on practice.", "positive"))
        elif training_hours < 25:
            insights.append(("Limited Training Hours", f"Only {training_hours} hours logged; may require supplementary lab exercises.", "negative"))

        if insights:
            for title, desc, tone in insights:
                if tone == "positive":
                    st.success(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.info("Metrics are within the standard range.")

        st.markdown(f"""
        <div class="recommendation-box">
            <strong style="color: #34D399;">Learning & Development Recommendation:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'Trainee is ready to transition to advanced electives or mentor peers.' if prediction == 'Successful' else 'Schedule targeted 1-on-1 tutoring, provide supplementary review materials, and monitor module attendance closely.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Transformed Parameters"):
        st.dataframe(sample_df, width="stretch")

# --- TAB 2: Batch Cohort Analysis ---
with tabs[1]:
    st.subheader("Batch Trainee Cohort Evaluation")
    st.write("Upload a CSV cohort file or examine the pre-loaded dataset.")

    csv_file = st.file_uploader("Upload Trainee Data CSV", type=["csv"], key="trainee_csv")
    df_cohort = None

    if csv_file is not None:
        df_cohort = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_cohort)} trainee records from file.")
    else:
        sample_path = get_asset_path("data/employee_training_success.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline training dataset (`data/employee_training_success.csv`)", value=True):
                df_cohort = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_cohort)} records from sample training dataset.")

    if df_cohort is not None:
        required_cols = ["age", "years_experience", "training_hours", "attendance_percent", "pre_training_score", "department"]
        missing = [c for c in required_cols if c not in df_cohort.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Run Cohort Analysis", type="primary"):
                with st.spinner("Evaluating cohort predictions..."):
                    batch_preds = model.predict(df_cohort[required_cols])
                    batch_probs = model.predict_proba(df_cohort[required_cols])
                    classes = list(model.classes_)
                    s_idx = classes.index("Successful") if "Successful" in classes else 1
                    success_probs = batch_probs[:, s_idx] * 100

                    res_df = df_cohort.copy()
                    res_df["Predicted_Result"] = batch_preds
                    res_df["Success_Probability_%"] = np.round(success_probs, 1)

                    success_count = sum(batch_preds == "Successful")
                    improve_count = sum(batch_preds == "Needs Improvement")
                    success_rate = (success_count / len(res_df)) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Trainees", len(res_df))
                    m2.metric("Successful", success_count, delta=f"{success_rate:.1f}%")
                    m3.metric("Needs Improvement", improve_count)
                    m4.metric("Average Success Prob", f"{np.mean(success_probs):.1f}%")

                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        filter_res = st.selectbox("Filter by Result:", ["All", "Successful", "Needs Improvement"])
                    with f_col2:
                        depts = ["All"] + sorted(list(df_cohort["department"].unique()))
                        filter_dept = st.selectbox("Filter by Department:", depts)

                    view_df = res_df
                    if filter_res != "All":
                        view_df = view_df[view_df["Predicted_Result"] == filter_res]
                    if filter_dept != "All":
                        view_df = view_df[view_df["department"] == filter_dept]

                    st.dataframe(view_df, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cohort Results as CSV",
                        data=csv_export,
                        file_name="employee_training_predictions.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Performance & Insights ---
with tabs[2]:
    st.subheader("Model Evaluation & Architecture")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        #### 🤖 Model Architecture & Pipeline
        - **Model**: `RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')`
        - **Pipeline Preprocessor**:
            - `OneHotEncoder` on categorical variable `department`
            - Passthrough on numerical attributes (`age`, `years_experience`, `training_hours`, `attendance_percent`, `pre_training_score`)
        - **Evaluation Performance**:
            - **Accuracy**: **90.00%** on Stratified Test Split
            - **Precision (Successful)**: ~0.95
            - **Recall (Needs Improvement)**: ~0.93
        """)

    with col_m2:
        cm_path = get_asset_path("confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Evaluation Confusion Matrix (Test Split)", width="stretch")
        else:
            st.info("Confusion matrix image not found.")

st.caption("Employee Training & Talent Development Suite • Scikit-learn & Streamlit")
