import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add a weather-themed background and header
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80') no-repeat center center fixed;
        background-size: cover;
    }
    .weather-header {
        color: #fff;
        text-shadow: 2px 2px 8px #00000099;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5em;
        text-align: center;
    }
    .weather-sub {
        color: #fff;
        text-shadow: 1px 1px 6px #00000099;
        font-size: 1.2rem;
        margin-bottom: 2em;
        text-align: center;
    }
    </style>
    <div class="weather-header">🌦️ Air Quality Prediction System</div>
    <div class="weather-sub">Upload your air quality dataset and get instant analysis, predictions, and visualizations.<br>Enjoy the weather effect while you explore your data!</div>
    """,
    unsafe_allow_html=True
)

# Add a weather animation (animated SVG cloud/rain)
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 2em;">
    <svg width="120" height="80" viewBox="0 0 120 80">
      <g>
        <ellipse cx="60" cy="50" rx="40" ry="20" fill="#fff" opacity="0.7"/>
        <ellipse cx="80" cy="55" rx="20" ry="12" fill="#fff" opacity="0.6"/>
        <ellipse cx="40" cy="55" rx="20" ry="12" fill="#fff" opacity="0.6"/>
        <line x1="50" y1="70" x2="50" y2="80" stroke="#4fc3f7" stroke-width="4">
          <animate attributeName="y1" values="70;80;70" dur="1s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="80;90;80" dur="1s" repeatCount="indefinite"/>
        </line>
        <line x1="70" y1="70" x2="70" y2="80" stroke="#4fc3f7" stroke-width="4">
          <animate attributeName="y1" values="70;80;70" dur="1.2s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="80;90;80" dur="1.2s" repeatCount="indefinite"/>
        </line>
      </g>
    </svg>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader('Upload your Air Quality CSV file', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')
    st.subheader('Data Preview')
    st.dataframe(df.head(), use_container_width=True)

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Target selection
    target_col = st.selectbox('Select target column for prediction', options=df.columns, index=list(df.columns).index('CO(GT)') if 'CO(GT)' in df.columns else 0)
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=['number'])

    # Interactive feature selection
    st.subheader('Feature Selection')
    selected_features = st.multiselect('Select features to use for prediction', options=X.columns, default=list(X.columns))
    X = X[selected_features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    st.subheader('Model Selection')
    model_type = st.radio('Choose a model', ['Random Forest', 'Decision Tree'])
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.subheader('Model Evaluation')
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    # ROC Curve (multiclass)
    st.subheader('ROC Curve (Multiclass)')
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test_scaled)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fig_roc, ax_roc = plt.subplots()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax_roc.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic Curve (Multiclass)')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    # EDA: Histogram
    st.subheader('EDA: Histogram of Target')
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(y, bins=20, color='#4fc3f7', edgecolor='black')
    ax_hist.set_title(f'{target_col} Distribution')
    ax_hist.set_xlabel(target_col)
    ax_hist.set_ylabel('Frequency')
    st.pyplot(fig_hist)

    # EDA: Boxplot for outlier detection
    st.subheader('EDA: Boxplot for Outlier Detection')
    y_numeric = pd.to_numeric(y, errors='coerce')
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=y_numeric, ax=ax_box, color='#81d4fa')
    ax_box.set_title(f'Boxplot for {target_col}')
    st.pyplot(fig_box)

    # GridSearchCV for Decision Tree
    st.subheader('Decision Tree with GridSearchCV')
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_train_scaled, y_train)
    st.write(f'Best parameters from GridSearch: {grid_search.best_params_}')

    # Prediction on new data
    st.subheader('Make Prediction')
    uploaded_pred = st.file_uploader('Upload a CSV for Prediction (same columns as training features)', type=['csv'], key='pred')
    if uploaded_pred:
        pred_df = pd.read_csv(uploaded_pred, delimiter=';')
        pred_X = pred_df[X.columns].fillna(X.median(numeric_only=True))
        pred_X = pred_X.select_dtypes(include=['number'])
        pred_X_scaled = scaler.transform(pred_X)
        pred_result = model.predict(pred_X_scaled)
        st.write('Predictions:')
        st.write(pred_result)

# Remove the auto-launch code to prevent infinite browser tabs
# if __name__ == "__main__":
#     import os
#     os.system('python -m streamlit run air_quality_app.py')
