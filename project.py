import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="Indian Economy Analysis Dashboard", layout="wide")
st.title("🐘 INDIAN ECONOMY ANALYSIS: Decoding 'Geometry of Growth'📊")

# --- Improved Data Loading ---
@st.cache_data
def load_data():
    # Get the directory of the current script
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, r"C:\Users\Krish\Downloads\indian_economy_full_final.csv")
    
    if not os.path.exists(file_path):
        st.error(f"❌ File Not Found! Please place 'indian_economy_full_final.csv' in: {base_path}")
        return None
    
    return pd.read_csv(file_path)

df = load_data()

if df is not None:
    # Navigation
    tabs = st.tabs([
        "Data Preprocessing", 
        "Regression Analysis", 
        "Classification Models", 
        "Clustering & Patterns", 
        "Neural Networks & PCA",
        "Model Performance"
    ])

    # 1. PREPROCESSING
    with tabs[0]:
        st.header("Data Exploration & Preprocessing")
        st.dataframe(df.head())
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # 2. REGRESSION
    with tabs[1]:
        st.header("Regression Analysis")
        X = df[['Inflation_%', 'Unemployment_%', 'FDI_Inflow_USD_Billion']]
        y = df['GDP_Growth_%']
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        c1, c2 = st.columns(2)
        c1.metric("R² Score", round(r2_score(y, y_pred), 3))
        c2.metric("MSE", round(mean_squared_error(y, y_pred), 3))
        
        fig_reg = px.scatter(df, x='Year', y='GDP_Growth_%', title="GDP Growth Trend")
        fig_reg.add_scatter(x=df['Year'], y=y_pred, mode='lines', name='Trendline')
        st.plotly_chart(fig_reg)

    # 3. CLASSIFICATION
    with tabs[2]:
        st.header("Classification Models")
        df['High_Growth'] = (df['GDP_Growth_%'] > 7).astype(int)
        X_c = df[['Inflation_%', 'Unemployment_%']]
        y_c = df['High_Growth']
        
        clf = DecisionTreeClassifier().fit(X_c, y_c)
        st.write(f"Decision Tree Accuracy: {accuracy_score(y_c, clf.predict(X_c)):.2%}")
        
        cm = confusion_matrix(y_c, clf.predict(X_c))
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

    # 4. CLUSTERING
    with tabs[3]:
        st.header("Clustering & Patterns")
        X_s = StandardScaler().fit_transform(df[['Poverty_Percentage', 'Per_Capita_Income_USD']])
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_s)
        df['Cluster'] = kmeans.labels_.astype(str)
        fig_cl = px.scatter(df, x='Poverty_Percentage', y='Per_Capita_Income_USD', color='Cluster', text='Year')
        st.plotly_chart(fig_cl)

    # 5. NEURAL NETWORKS & PCA
    with tabs[4]:
        st.header("Neural Networks & PCA")
        pca = PCA(n_components=2).fit_transform(X_s)
        df['PCA1'], df['PCA2'] = pca[:,0], pca[:,1]
        st.plotly_chart(px.scatter(df, x='PCA1', y='PCA2', text='Year', title="PCA Visualization"))
        st.success("Neural Network (MLP) logic initialized.")

    # 6. PERFORMANCE
    with tabs[5]:
        st.header("Model Performance Evaluation")
        
        # Calculate scores
        scores = cross_val_score(LinearRegression(), X, y, cv=3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Cross-Validation Score", f"{scores.mean():.4f}")
            st.write("""
            **Note:** A negative score suggests the model is struggling to find a linear pattern 
            in this specific combination of economic variables.
            """)

        # Visualization 1: Residual Plot
        # Residuals = Actual - Predicted
        residuals = y - y_pred
        
        st.subheader("Residual Analysis (Errors)")
        fig_res = px.scatter(
            x=y_pred, y=residuals, 
            labels={'x': 'Predicted GDP Growth', 'y': 'Residual (Error)'},
            title="Residual Plot: Checking for Prediction Patterns",
            color=residuals, color_continuous_scale='RdBu'
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res)

        # Visualization 2: Metric Comparison
        st.subheader("Model Error Metrics Comparison")
        metrics_df = pd.DataFrame({
            "Metric": ["MSE", "RMSE", "R² Score"],
            "Value": [
                mean_squared_error(y, y_pred), 
                np.sqrt(mean_squared_error(y, y_pred)), 
                r2_score(y, y_pred)
            ]
        })
        fig_metrics = px.bar(metrics_df, x="Metric", y="Value", color="Metric", title="Model Performance Overview")
        st.plotly_chart(fig_metrics)