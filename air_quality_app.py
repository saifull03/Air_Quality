# Final fix: Ensuring all texts are fully visible and contrast is high over the background
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config early
st.set_page_config(page_title="Air Quality Predictor", layout="wide")

# Enhanced high-contrast styling
st.markdown("""
<style>
body {
    background: linear-gradient(to top right, rgba(255, 248, 252, 0.9), rgba(255, 245, 250, 0.9));
    background-attachment: fixed;
}
.stApp {
    background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)),
                url('https://images.unsplash.com/photo-1561484932-07c029bbcaec?auto=format&fit=crop&w=1500&q=80')
                no-repeat center center fixed;
    background-size: cover;
    color: #111 !important;
}
.weather-banner {
    display: flex;
    justify-content: center;
    margin-bottom: 2em;
}
.weather-banner-bg {
    background: #ffffffee;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    max-width: 800px;
    width: 100%;
    text-align: center;
}
.weather-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #222;
}
.weather-sub {
    font-size: 1.15rem;
    color: #333;
    margin-top: 0.5rem;
}
label, .stTextInput, .stSelectbox, .stRadio, .stMultiSelect {
    color: #111 !important;
}
.stTabs [role="tab"] {
    color: #111 !important;
    font-weight: 600;
    background: rgba(255,255,255,0.9);
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
    margin-right: 5px;
}
</style>
<div class="weather-banner">
    <div class="weather-banner-bg">
        <div class="weather-header">🌸 Air Quality Prediction System</div>
        <div class="weather-sub">Upload your dataset to visualize, analyze, and predict air quality with an elegant soft UI experience.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Upload file ---- #
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

uploaded_file = st.file_uploader("📁 Upload your Air Quality CSV file", type=["csv"])
if uploaded_file:
    st.session_state.uploaded_data = uploaded_file

if st.session_state.uploaded_data:
    df = pd.read_csv(st.session_state.uploaded_data, delimiter=';')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox("🎯 Target Column", options=df.columns, index=df.columns.get_loc("CO(GT)") if "CO(GT)" in df.columns else 0)
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include='number').fillna(df.median(numeric_only=True))

    with st.expander("🧮 Select Features"):
        selected_features = st.multiselect("Choose input features:", options=X.columns, default=list(X.columns))
        X = X[selected_features]

    # ---- Model training ---- #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("⚙️ Model Selection")
    model_type = st.radio("Select a model:", ["Random Forest", "Decision Tree"])

    with st.spinner("Training model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42) if model_type == 'Random Forest' else DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    st.subheader("📊 Model Evaluation")
    tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🧱 Confusion Matrix", "📉 ROC Curve"])

    with tab1:
        st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**Precision**: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"**Recall**: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"**F1 Score**: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    with tab2:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with tab3:
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_score = model.predict_proba(X_test_scaled)
        fig, ax = plt.subplots()
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.subheader("🔍 Exploratory Data Analysis")
    with st.expander("📊 Target Histogram"):
        fig, ax = plt.subplots()
        ax.hist(y, bins=20, color='#4fc3f7', edgecolor='black')
        ax.set_title(f"Distribution of {target_col}")
        st.pyplot(fig)

    with st.expander("📦 Outlier Boxplot"):
        fig, ax = plt.subplots()
        sns.boxplot(x=pd.to_numeric(y, errors='coerce'), ax=ax, color='#81d4fa')
        ax.set_title(f"Boxplot for {target_col}")
        st.pyplot(fig)

    st.subheader("🧪 Decision Tree GridSearchCV")
    with st.spinner("Running grid search..."):
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        st.write("Best parameters:", grid.best_params_)

    st.subheader("📥 Predict on New Data")
    if "uploaded_pred_file" not in st.session_state:
        st.session_state.uploaded_pred_file = None

    pred_file = st.file_uploader("Upload new CSV for prediction", type=["csv"], key="pred")
    if pred_file:
        st.session_state.uploaded_pred_file = pred_file

    if st.session_state.uploaded_pred_file:
        pred_df = pd.read_csv(st.session_state.uploaded_pred_file, delimiter=';')
        pred_X = pred_df[selected_features].fillna(X.median(numeric_only=True))
        pred_X_scaled = scaler.transform(pred_X)
        preds = model.predict(pred_X_scaled)
        st.write("📌 Predictions:")
        st.write(preds)
