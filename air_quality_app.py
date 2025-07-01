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

st.title('Air Quality Prediction System')

uploaded_file = st.file_uploader('Upload your Air Quality CSV file', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')
    st.subheader('Data Preview')
    st.write(df.head())

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Target selection
    target_col = st.selectbox('Select target column for prediction', options=df.columns, index=list(df.columns).index('CO(GT)') if 'CO(GT)' in df.columns else 0)
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=['number'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    ax_hist.hist(y, bins=20)
    ax_hist.set_title(f'{target_col} Distribution')
    ax_hist.set_xlabel(target_col)
    ax_hist.set_ylabel('Frequency')
    st.pyplot(fig_hist)

    # EDA: Boxplot for outlier detection
    st.subheader('EDA: Boxplot for Outlier Detection')
    y_numeric = pd.to_numeric(y, errors='coerce')
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=y_numeric, ax=ax_box)
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
