
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --- App Header ---
st.set_page_config(page_title="Linear Regression Explorer", layout="wide")
st.title("Linear Regression Demonstrator")
st.write("""
This app demonstrates linear regression on two datasets:
1.  **Synthetic Data:** Explore how parameters like slope, noise, and sample size affect the model.
2.  **Iris Dataset:** Predict one of the Iris measurements based on the other three.

Use the tabs below to switch between the two demonstrations.
""")

# --- Helper function to convert dataframe to CSV for download ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Synthetic ax-b", "Iris Regression"])

# --- Part A: Synthetic Linear Data ---
with tab1:
    st.header("Part A: Linear Regression on Synthetic Data")

    # --- User Controls ---
    st.sidebar.header("Synthetic Data Controls")
    a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.5, step=0.1)
    var = st.sidebar.slider("Noise Variance (var)", min_value=0, max_value=5000, value=100, step=50)
    n = st.sidebar.slider("Number of points (n)", min_value=20, max_value=1000, value=200, step=10)
    seed = st.sidebar.number_input("Random Seed", value=42)

    # --- Data Generation ---
    b = 30
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    y_true = a * x - b
    noise = np.random.normal(0, np.sqrt(var), n)
    y = y_true + noise
    
    synthetic_data = pd.DataFrame({'x': x, 'y': y})

    # --- Modeling ---
    X = synthetic_data[['x']]
    y_data = synthetic_data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=seed)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    y_hat_full = model.predict(X)

    # --- Metrics ---
    learned_slope = model.coef_[0]
    learned_intercept = model.intercept_
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learned Slope", f"{learned_slope:.2f}")
    col2.metric("Learned Intercept", f"{learned_intercept:.2f}")
    col3.metric("R² Score", f"{r2:.3f}")
    col4.metric("RMSE", f"{rmse:.2f}")

    # --- Plotting ---
    st.subheader("Data and Regression Fit")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of all points
    ax.scatter(x, y, alpha=0.6, label="Generated Data Points")
    
    # Regression line
    ax.plot(x, y_hat_full, color='red', linewidth=2, label="Fitted Regression Line")
    
    # --- Outlier Identification ---
    residuals = np.abs(y_data - y_hat_full)
    outlier_indices = residuals.nlargest(5).index
    
    st.write("Top 5 Outliers (by absolute residual):")
    for i, idx in enumerate(outlier_indices):
        ax.annotate(f"#{i+1}", (X.loc[idx, 'x'], y_data[idx]), 
                    xytext=(X.loc[idx, 'x'] + 0.2, y_data[idx]),
                    fontweight='bold', color='darkorange')
        ax.scatter(X.loc[idx, 'x'], y_data[idx], facecolors='none', edgecolors='darkorange', s=100, linewidths=1.5)


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    title = f"y = {a:.2f}x - {b} + N(0, {var}); n={n}\n"
    title += f"Fit: y = {learned_slope:.2f}x + {learned_intercept:.2f}; R²={r2:.3f}, RMSE={rmse:.2f}"
    ax.set_title(title)
    st.pyplot(fig)

    # --- Data Export ---
    st.subheader("Export Data")
    export_df = pd.DataFrame({
        'x': x,
        'y_observed': y,
        'y_true (no noise)': y_true,
        'y_hat (prediction)': y_hat_full,
        'residual': y - y_hat_full
    })
    csv_export = convert_df_to_csv(export_df)
    st.download_button(
        label="Download Dataset as CSV",
        data=csv_export,
        file_name=f"synthetic_data_a{a}_var{var}_n{n}.csv",
        mime="text/csv",
    )


# --- Part B: Iris Dataset Regression ---
with tab2:
    st.header("Part B: Linear Regression on the Iris Dataset")

    # --- Data Loading and User Controls ---
    iris = load_iris(as_frame=True)
    iris_df = iris.frame.copy()
    iris_df.rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width'
    }, inplace=True)
    
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    target_variable = st.selectbox(
        "Choose the target variable to predict:",
        options=numeric_cols,
        index=2 # Default to petal_length
    )
    
    feature_variables = [col for col in numeric_cols if col != target_variable]
    
    st.write(f"**Target:** `{target_variable}`")
    st.write(f"**Features:** `{', '.join(feature_variables)}`")

    # --- Modeling ---
    X_iris = iris_df[feature_variables]
    y_iris = iris_df[target_variable]
    
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train_iris, y_train_iris)
    y_pred_iris_test = pipeline.predict(X_test_iris)

    # --- Metrics ---
    st.subheader("Model Performance")
    r2_iris = r2_score(y_test_iris, y_pred_iris_test)
    rmse_iris = np.sqrt(mean_squared_error(y_test_iris, y_pred_iris_test))
    
    col1_iris, col2_iris = st.columns(2)
    col1_iris.metric("R² Score (on test set)", f"{r2_iris:.4f}")
    col2_iris.metric("RMSE (on test set)", f"{rmse_iris:.4f}")

    st.subheader("Learned Coefficients")
    coefficients = pipeline.named_steps['regressor'].coef_
    intercept = pipeline.named_steps['regressor'].intercept_
    
    coef_df = pd.DataFrame(coefficients, index=feature_variables, columns=['Coefficient'])
    coef_df.loc['Intercept'] = intercept
    st.table(coef_df)

    # --- Plotting ---
    st.subheader("Diagnostic Plots")
    
    # Predicted vs. Actual
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test_iris, y_pred_iris_test, alpha=0.7)
    ax1.plot([y_test_iris.min(), y_test_iris.max()], [y_test_iris.min(), y_test_iris.max()], 'k--', lw=2, label="Ideal Fit")
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title("Predicted vs. Actual Values")
    ax1.legend()
    
    # Residuals vs. Predicted
    residuals_iris = y_test_iris - y_pred_iris_test
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_pred_iris_test, residuals_iris, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', label="Zero Residual")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals (Actual - Predicted)")
    ax2.set_title("Residuals vs. Predicted Values")
    ax2.legend()

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.pyplot(fig1)
    with plot_col2:
        st.pyplot(fig2)

    # --- Data Export ---
    st.subheader("Export Test Predictions")
    test_predictions_df = X_test_iris.copy()
    test_predictions_df['actual_' + target_variable] = y_test_iris
    test_predictions_df['predicted_' + target_variable] = y_pred_iris_test
    test_predictions_df['residual'] = residuals_iris
    
    csv_iris_export = convert_df_to_csv(test_predictions_df)
    st.download_button(
        label="Download Test Predictions as CSV",
        data=csv_iris_export,
        file_name=f"iris_test_predictions_{target_variable}.csv",
        mime="text/csv",
    )
