# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # from sklearn.metrics import (
# #     accuracy_score,
# #     precision_score,
# #     recall_score,
# #     f1_score,
# #     confusion_matrix,
# #     roc_auc_score
# # )

# # # -------------------- PAGE CONFIG --------------------
# # st.set_page_config(
# #     page_title="Customer Churn Analytics",
# #     page_icon="📊",
# #     layout="wide"
# # )

# # # -------------------- CUSTOM CSS --------------------
# # st.markdown("""
# # <style>

# # /* App background */
# # .stApp {
# #     background-color: #f4f6fb;
# # }

# # /* Main title */
# # .main-title {
# #     font-size: 44px;
# #     font-weight: 800;
# #     color: #1f2937;
# #     text-align: center;
# #     margin-bottom: 4px;
# # }

# # /* Subtitle */
# # .subtitle {
# #     text-align: center;
# #     color: #4b5563;
# #     font-size: 16px;
# #     margin-bottom: 40px;
# # }

# # /* Card base */
# # .card {
# #     background-color: #ffffff;
# #     padding: 24px;
# #     border-radius: 16px;
# #     border: 1px solid #e5e7eb;
# #     box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
# #     text-align: center;
# # }

# # /* KPI value */
# # .kpi-value {
# #     font-size: 34px;
# #     font-weight: 800;
# #     color: #111827;
# # }

# # /* KPI label */
# # .kpi-label {
# #     margin-top: 6px;
# #     font-size: 14px;
# #     font-weight: 500;
# #     color: #6b7280;
# # }

# # /* Accent colors */
# # .kpi-red { color: #dc2626; }
# # .kpi-green { color: #16a34a; }
# # .kpi-blue { color: #2563eb; }

# # /* Section titles */
# # .section-title {
# #     font-size: 26px;
# #     font-weight: 700;
# #     color: #1f2937;
# #     margin-top: 50px;
# #     margin-bottom: 20px;
# # }

# # /* Divider */
# # hr {
# #     border: none;
# #     height: 1px;
# #     background-color: #e5e7eb;
# #     margin: 40px 0;
# # }

# # </style>
# # """, unsafe_allow_html=True)


# # # -------------------- LOAD MODEL --------------------
# # model = joblib.load("churn_model.pkl")

# # # -------------------- LOAD DATA --------------------
# # df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
# # df.drop(columns=["customerID"], inplace=True)
# # df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# # df.dropna(inplace=True)

# # X = df.drop("Churn", axis=1)
# # y = df["Churn"].map({"Yes": 1, "No": 0})

# # # -------------------- PREDICTIONS --------------------
# # y_prob = model.predict_proba(X)[:, 1]
# # y_pred = (y_prob >= 0.5).astype(int)

# # # -------------------- METRICS --------------------
# # accuracy = accuracy_score(y, y_pred)
# # precision = precision_score(y, y_pred)
# # recall = recall_score(y, y_pred)
# # f1 = f1_score(y, y_pred)
# # roc_auc = roc_auc_score(y, y_prob)

# # cm = confusion_matrix(y, y_pred)
# # tn, fp, fn, tp = cm.ravel()

# # total_users = len(df)
# # churn_users = y.sum()
# # non_churn_users = total_users - churn_users

# # # -------------------- HEADER --------------------
# # st.markdown('<p class="main-title">Customer Churn Analytics</p>', unsafe_allow_html=True)
# # st.markdown('<p class="subtitle">Executive dashboard for retention intelligence</p>', unsafe_allow_html=True)

# # # -------------------- KPI CARDS --------------------
# # c1, c2, c3, c4 = st.columns(4)

# # with c1:
# #     st.markdown(f"""
# #     <div class="card">
# #         <div class="kpi-value">{total_users}</div>
# #         <div class="kpi-label">Total Customers</div>
# #     </div>
# #     """, unsafe_allow_html=True)

# # with c2:
# #     st.markdown(f"""
# #     <div class="card">
# #         <div class="kpi-value kpi-red">{churn_users}</div>
# #         <div class="kpi-label">Likely to Leave</div>
# #     </div>
# #     """, unsafe_allow_html=True)

# # with c3:
# #     st.markdown(f"""
# #     <div class="card">
# #         <div class="kpi-value kpi-green">{non_churn_users}</div>
# #         <div class="kpi-label">Likely to Stay</div>
# #     </div>
# #     """, unsafe_allow_html=True)

# # with c4:
# #     st.markdown(f"""
# #     <div class="card">
# #         <div class="kpi-value kpi-blue">{accuracy:.2%}</div>
# #         <div class="kpi-label">Model Accuracy</div>
# #     </div>
# #     """, unsafe_allow_html=True)


# # # -------------------- PERFORMANCE METRICS --------------------
# # st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

# # m1, m2, m3, m4, m5 = st.columns(5)

# # metrics = [
# #     ("Precision", precision),
# #     ("Recall (Churn)", recall),
# #     ("F1 Score", f1),
# #     ("ROC AUC", roc_auc),
# #     ("Error Rate", 1 - accuracy)
# # ]

# # for col, (label, value) in zip([m1, m2, m3, m4, m5], metrics):
# #     col.markdown(f"""
# #     <div class="card">
# #         <div class="kpi-value">{value:.2f}</div>
# #         <div class="kpi-label">{label}</div>
# #     </div>
# #     """, unsafe_allow_html=True)


# # # -------------------- CONFUSION MATRIX --------------------
# # st.markdown('<div class="section-title">Prediction Quality</div>', unsafe_allow_html=True)

# # left, right = st.columns([3, 1])

# # with right:
# #     fig, ax = plt.subplots(figsize=(3.2, 3.2))
# #     sns.heatmap(
# #         cm,
# #         annot=True,
# #         fmt="d",
# #         cmap="Blues",
# #         cbar=False,
# #         xticklabels=["Not Churn", "Churn"],
# #         yticklabels=["Not Churn", "Churn"],
# #         annot_kws={"size": 11},
# #         ax=ax
# #     )
# #     ax.set_xlabel("Predicted", fontsize=10)
# #     ax.set_ylabel("Actual", fontsize=10)
# #     st.pyplot(fig)

# # with left:
# #     st.markdown(f"""
# # <div class="card">
# # <b>Executive Summary</b><br><br>

# # • The model successfully identified <b>{tp}</b> high-risk customers who are likely to churn, enabling targeted retention actions.<br>
# # • <b>{fp}</b> customers were conservatively flagged as churn risks, which is acceptable for proactive campaigns.<br>
# # • <b>{fn}</b> churn cases were missed, representing potential revenue leakage and an opportunity for further optimization.<br><br>

# # <b>Business Strategy:</b><br>
# # The model prioritizes <b>Recall</b> to minimize customer loss, even if it slightly increases false alerts — a standard industry practice in churn management.
# # </div>
# # """, unsafe_allow_html=True)


# # # -------------------- FOOTER --------------------
# # st.markdown("""
# # <hr>
# # <p style="text-align:center;color:#9ca3af;">
# # Customer Retention Intelligence • Production-Ready ML Dashboard
# # </p>
# # """, unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     roc_auc_score
# )

# # -------------------------------------------------
# # PAGE CONFIG
# # -------------------------------------------------
# st.set_page_config(
#     page_title="Customer Churn Analytics",
#     page_icon="📊",
#     layout="wide"
# )

# # -------------------------------------------------
# # CUSTOM CSS (MODERN, HIGH-CONTRAST)
# # -------------------------------------------------
# st.markdown("""
# <style>
# .stApp {
#     background-color: #f4f6fb;
# }
# .main-title {
#     font-size: 44px;
#     font-weight: 800;
#     color: #1f2937;
#     text-align: center;
# }
# .subtitle {
#     text-align: center;
#     color: #4b5563;
#     font-size: 16px;
#     margin-bottom: 40px;
# }
# .card {
#     background-color: #ffffff;
#     padding: 24px;
#     border-radius: 16px;
#     border: 1px solid #e5e7eb;
#     box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
#     text-align: center;
# }
# .kpi-value {
#     font-size: 34px;
#     font-weight: 800;
#     color: #111827;
# }
# .kpi-label {
#     margin-top: 6px;
#     font-size: 14px;
#     font-weight: 500;
#     color: #6b7280;
# }
# .kpi-red { color: #dc2626; }
# .kpi-green { color: #16a34a; }
# .kpi-blue { color: #2563eb; }
# .section-title {
#     font-size: 26px;
#     font-weight: 700;
#     color: #1f2937;
#     margin-top: 50px;
#     margin-bottom: 20px;
# }
# hr {
#     border: none;
#     height: 1px;
#     background-color: #e5e7eb;
#     margin: 40px 0;
# }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------------------------
# # LOAD MODEL
# # -------------------------------------------------
# model = joblib.load("churn_model.pkl")

# # -------------------------------------------------
# # LOAD & PREPARE DATA
# # -------------------------------------------------
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
# df.drop(columns=["customerID"], inplace=True)
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# df.dropna(inplace=True)

# X = df.drop("Churn", axis=1)
# y = df["Churn"].map({"Yes": 1, "No": 0})

# # -------------------------------------------------
# # PREDICTIONS & METRICS (GLOBAL)
# # -------------------------------------------------
# y_prob = model.predict_proba(X)[:, 1]
# y_pred = (y_prob >= 0.5).astype(int)

# accuracy = accuracy_score(y, y_pred)
# precision = precision_score(y, y_pred)
# recall = recall_score(y, y_pred)
# f1 = f1_score(y, y_pred)
# roc_auc = roc_auc_score(y, y_prob)

# cm = confusion_matrix(y, y_pred)
# tn, fp, fn, tp = cm.ravel()

# total_users = len(df)
# churn_users = y.sum()
# non_churn_users = total_users - churn_users

# # -------------------------------------------------
# # HEADER
# # -------------------------------------------------
# st.markdown('<div class="main-title">Customer Churn Analytics</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Executive dashboard for customer retention intelligence</div>', unsafe_allow_html=True)

# # -------------------------------------------------
# # KPI CARDS
# # -------------------------------------------------
# c1, c2, c3, c4 = st.columns(4)

# with c1:
#     st.markdown(f"""
#     <div class="card">
#         <div class="kpi-value">{total_users}</div>
#         <div class="kpi-label">Total Customers</div>
#     </div>
#     """, unsafe_allow_html=True)

# with c2:
#     st.markdown(f"""
#     <div class="card">
#         <div class="kpi-value kpi-red">{churn_users}</div>
#         <div class="kpi-label">Likely to Leave</div>
#     </div>
#     """, unsafe_allow_html=True)

# with c3:
#     st.markdown(f"""
#     <div class="card">
#         <div class="kpi-value kpi-green">{non_churn_users}</div>
#         <div class="kpi-label">Likely to Stay</div>
#     </div>
#     """, unsafe_allow_html=True)

# with c4:
#     st.markdown(f"""
#     <div class="card">
#         <div class="kpi-value kpi-blue">{accuracy:.2%}</div>
#         <div class="kpi-label">Model Accuracy</div>
#     </div>
#     """, unsafe_allow_html=True)

# # -------------------------------------------------
# # PERFORMANCE METRICS
# # -------------------------------------------------
# st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

# m1, m2, m3, m4, m5 = st.columns(5)

# metrics = [
#     ("Precision", precision),
#     ("Recall (Churn)", recall),
#     ("F1 Score", f1),
#     ("ROC AUC", roc_auc),
#     ("Error Rate", 1 - accuracy)
# ]

# for col, (label, value) in zip([m1, m2, m3, m4, m5], metrics):
#     col.markdown(f"""
#     <div class="card">
#         <div class="kpi-value">{value:.2f}</div>
#         <div class="kpi-label">{label}</div>
#     </div>
#     """, unsafe_allow_html=True)

# # -------------------------------------------------
# # CONFUSION MATRIX + EXECUTIVE SUMMARY
# # -------------------------------------------------
# st.markdown('<div class="section-title">Prediction Quality</div>', unsafe_allow_html=True)

# left, right = st.columns([3, 1])

# with right:
#     fig, ax = plt.subplots(figsize=(3.2, 3.2))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         cbar=False,
#         xticklabels=["Not Churn", "Churn"],
#         yticklabels=["Not Churn", "Churn"],
#         annot_kws={"size": 11},
#         ax=ax
#     )
#     ax.set_xlabel("Predicted", fontsize=10)
#     ax.set_ylabel("Actual", fontsize=10)
#     st.pyplot(fig)

# with left:
#     st.markdown(f"""
#     <div class="card">
#     <b>Executive Summary</b><br><br>
#     • <b>{tp}</b> high-risk customers were accurately identified for retention action.<br>
#     • <b>{fp}</b> customers were conservatively flagged, acceptable for proactive campaigns.<br>
#     • <b>{fn}</b> churn cases were missed, indicating scope for threshold tuning.<br><br>
#     <b>Strategy:</b> The model prioritizes <b>Recall</b> to minimize revenue loss, aligning with industry churn-management best practices.
#     </div>
#     """, unsafe_allow_html=True)

# # -------------------------------------------------
# # SIDEBAR — USER INPUT SIMULATOR
# # -------------------------------------------------
# st.sidebar.title("🔧 Customer Risk Simulator")
# st.sidebar.write("Modify customer attributes to simulate churn risk.")

# gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
# SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
# Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
# Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
# tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
# MonthlyCharges = st.sidebar.slider("Monthly Charges", 20.0, 120.0, 70.0)
# TotalCharges = st.sidebar.slider("Total Charges", 0.0, 9000.0, 2000.0)
# Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
# PaymentMethod = st.sidebar.selectbox(
#     "Payment Method",
#     [
#         "Electronic check",
#         "Mailed check",
#         "Bank transfer (automatic)",
#         "Credit card (automatic)"
#     ]
# )

# user_input = pd.DataFrame({
#     "gender": [gender],
#     "SeniorCitizen": [SeniorCitizen],
#     "Partner": [Partner],
#     "Dependents": [Dependents],
#     "tenure": [tenure],
#     "MonthlyCharges": [MonthlyCharges],
#     "TotalCharges": [TotalCharges],
#     "Contract": [Contract],
#     "PaymentMethod": [PaymentMethod]
# })

# user_prob = model.predict_proba(user_input)[0][1]
# user_pred = "Likely to Churn" if user_prob >= 0.5 else "Likely to Stay"
# risk_color = "#dc2626" if user_prob >= 0.5 else "#16a34a"

# # -------------------------------------------------
# # USER SIMULATION RESULT
# # -------------------------------------------------
# st.markdown('<div class="section-title">Customer Risk Simulation</div>', unsafe_allow_html=True)

# st.markdown(f"""
# <div class="card">
#     <div style="font-size:30px;font-weight:800;color:{risk_color};">
#         {user_pred}
#     </div>
#     <div style="font-size:18px;margin-top:10px;">
#         Churn Probability: <b>{user_prob:.2%}</b>
#     </div>
#     <div style="color:#6b7280;margin-top:10px;">
#         Adjust customer attributes from the left panel to observe real-time risk changes.
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # -------------------------------------------------
# # FOOTER
# # -------------------------------------------------
# st.markdown("""
# <hr>
# <p style="text-align:center;color:#9ca3af;">
# Customer Retention Intelligence • Production-Ready ML Dashboard
# </p>
# """, unsafe_allow_html=True)



import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (MODERN, HIGH-CONTRAST)
# -------------------------------------------------
# -------------------------------------------------
# CUSTOM CSS (MODERN, HIGH-CONTRAST)
# -------------------------------------------------
st.markdown("""
<style>
    /* 1. Main Background */
    .stApp { background-color: #f4f6fb; }

    /* 2. Main Title */
    .main-title {
        font-size: 44px; font-weight: 800;
        color: #1f2937; text-align: center;
    }

    /* 3. Subtitle */
    .subtitle {
        text-align: center; color: #4b5563;
        font-size: 16px; margin-bottom: 40px;
    }

    /* 4. Card Styling */
    .card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
        text-align: center;
    }

    /* ------------------------------------------------------- */
    /* ⚠️ THE OVERRIDE: FIX FOR INVISIBLE TEXT ⚠️              */
    /* ------------------------------------------------------- */
    
    /* Target every single element inside the colored boxes */
    div[data-testid="stNotification"], 
    div[data-testid="stNotification"] p, 
    div[data-testid="stNotification"] div, 
    div[data-testid="stNotification"] span, 
    div[data-testid="stNotification"] li {
        color: #1f2937 !important; /* Force text Dark Grey */
        font-weight: 700 !important; /* Force text Bold */
    }
    
    /* ------------------------------------------------------- */

    /* 5. KPI Styling */
    .kpi-value { font-size: 34px; font-weight: 800; color: #111827; }
    .kpi-label { font-size: 14px; color: #6b7280; }
    .kpi-red { color: #dc2626; }
    .kpi-green { color: #16a34a; }
    .kpi-blue { color: #2563eb; }
    
    /* 6. Section Titles */
    .section-title {
        font-size: 26px; font-weight: 700;
        color: #1f2937; margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("churn_model.pkl")

# -------------------------------------------------
# LOAD & PREPARE DATA
# -------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
df.drop(columns=["customerID"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# -------------------------------------------------
# GLOBAL METRICS
# -------------------------------------------------
y_prob_all = model.predict_proba(X)[:, 1]
y_pred_all = (y_prob_all >= 0.5).astype(int)

accuracy = accuracy_score(y, y_pred_all)
precision = precision_score(y, y_pred_all)
recall = recall_score(y, y_pred_all)
f1 = f1_score(y, y_pred_all)
roc_auc = roc_auc_score(y, y_prob_all)

cm = confusion_matrix(y, y_pred_all)
tn, fp, fn, tp = cm.ravel()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="main-title">Customer Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Executive dashboard for customer retention intelligence</div>', unsafe_allow_html=True)

# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='card'><div class='kpi-value'>{len(df)}</div><div class='kpi-label'>Total Customers</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div class='kpi-value kpi-red'>{y.sum()}</div><div class='kpi-label'>Likely to Leave</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='kpi-value kpi-green'>{len(df)-y.sum()}</div><div class='kpi-label'>Likely to Stay</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><div class='kpi-value kpi-blue'>{accuracy:.2%}</div><div class='kpi-label'>Model Accuracy</div></div>", unsafe_allow_html=True)

# -------------------------------------------------
# PERFORMANCE METRICS
# -------------------------------------------------
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [("Precision", precision), ("Recall", recall),
           ("F1 Score", f1), ("ROC-AUC", roc_auc),
           ("Error Rate", 1 - accuracy)]

for col, (label, val) in zip([m1, m2, m3, m4, m5], metrics):
    col.markdown(f"<div class='card'><div class='kpi-value'>{val:.2f}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
st.markdown('<div class="section-title">Prediction Quality</div>', unsafe_allow_html=True)

left, right = st.columns([3, 1])

with right:
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                cbar=False,
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                ax=ax)
    st.pyplot(fig)
    
with left:
    st.subheader("Executive Summary")
    
    # --- HARDCODED HTML ALERTS (100% VISIBLE FIX) ---
    
    # 1. Green Box (Success)
    st.markdown(f"""
    <div style="background-color:#dcfce7; padding:15px; border-radius:10px; margin-bottom:10px; border:1px solid #86efac;">
        <p style="color:#14532d; font-weight:700; font-size:16px; margin:0;">
            ✅ {tp} churn customers correctly identified
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Yellow Box (Warning)
    st.markdown(f"""
    <div style="background-color:#fef9c3; padding:15px; border-radius:10px; margin-bottom:10px; border:1px solid #fde047;">
        <p style="color:#713f12; font-weight:700; font-size:16px; margin:0;">
            ⚠️ {fp} customers conservatively flagged
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Red Box (Error)
    st.markdown(f"""
    <div style="background-color:#fee2e2; padding:15px; border-radius:10px; margin-bottom:10px; border:1px solid #fca5a5;">
        <p style="color:#7f1d1d; font-weight:700; font-size:16px; margin:0;">
            🛑 {fn} churn cases missed
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4. Blue Box (Info)
    st.markdown("""
    <div style="background-color:#dbeafe; padding:15px; border-radius:10px; margin-bottom:10px; border:1px solid #93c5fd;">
        <p style="color:#1e3a8a; font-weight:600; font-size:14px; margin:0;">
            ℹ️ <b>Strategy:</b> Recall is prioritized to minimize revenue loss 
            and ensure proactive customer retention.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR — MAIN FEATURES ONLY
# -------------------------------------------------
st.sidebar.title("🔧 Customer Risk Simulator")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges", 20.0, 120.0, 70.0)
TotalCharges = st.sidebar.slider("Total Charges", 0.0, 9000.0, 2000.0)
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check",
     "Bank transfer (automatic)","Credit card (automatic)"]
)

# -------------------------------------------------
# DEFAULTS FOR NON-MAIN FEATURES (HIDDEN)
# -------------------------------------------------
user_input = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
    "Contract": [Contract],
    "PaymentMethod": [PaymentMethod],

    # auto-filled defaults
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "PaperlessBilling": ["Yes"]
})

# -------------------------------------------------
# USER PREDICTION
# -------------------------------------------------
user_prob = model.predict_proba(user_input)[0][1]
user_pred = "Likely to Churn" if user_prob >= 0.5 else "Likely to Stay"
risk_color = "#dc2626" if user_prob >= 0.5 else "#16a34a"

# -------------------------------------------------
# USER RESULT
# -------------------------------------------------
# -------------------------------------------------
# USER RESULT
# -------------------------------------------------
st.markdown('<div class="section-title">Customer Risk Simulation</div>', unsafe_allow_html=True)

# Define dynamic content based on risk
if user_prob >= 0.7:
    risk_level = "Critical"
    risk_msg = "🚨 High Risk: Immediate retention offer required."
    action_item = "Suggest switching to 1-Year Contract with 15% discount."
    bar_color = "#dc2626"  # Red
elif user_prob >= 0.5:
    risk_level = "Moderate"
    risk_msg = "⚠️ At Risk: Customer showing signs of leaving."
    action_item = "Schedule a customer success check-in call."
    bar_color = "#f59e0b"  # Orange
else:
    risk_level = "Safe"
    risk_msg = "✅ Loyal: Customer is stable."
    action_item = "Good candidate for Upselling (e.g., Streaming TV)."
    bar_color = "#16a34a"  # Green

# Create columns for layout
r1, r2 = st.columns([1, 2])

# Left Column: The Big Score Card
with r1:
    st.markdown(f"""
    <div class="card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size:20px; color:#6b7280; margin-bottom: 10px;">Churn Probability</div>
        <div style="font-size:48px; font-weight:800; color:{risk_color};">
            {user_prob:.1%}
        </div>
        <div style="font-size:18px; font-weight:600; color:{risk_color}; margin-top:5px;">
            {user_pred}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Right Column: Strategy & Visuals

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center;color:#9ca3af;">
Customer Retention Intelligence • Production-Ready ML Dashboard
</p>
""", unsafe_allow_html=True)
